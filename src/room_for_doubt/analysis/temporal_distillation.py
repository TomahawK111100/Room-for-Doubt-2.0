"""Separate prepare / fit / final-evaluate stages for the residual feasibility pilot."""

import json
from pathlib import Path
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torchvision.datasets import CIFAR10, CIFAR100
from room_for_doubt.data.animal10n import Animal10N, verify_identity
from room_for_doubt.data.mini_webvision import MiniWebVision
from room_for_doubt.data.cifar_noisy import load_observed_labels
from room_for_doubt.data.kaggle_artifacts import load_completed_run, restore_base
from room_for_doubt.data.temporal_evidence import load_evidence, seal_evidence
from room_for_doubt.inference.temporal_evidence import extract
from room_for_doubt.training.temporal_snapshots import verify_snapshots, file_digest
from room_for_doubt.training.temporal_residual import fit, load_frozen, residuals
from room_for_doubt.evaluation.temporal_residual import (
    corrected_probabilities,
    evaluate_probabilities,
)
from room_for_doubt.evaluation.selection import paired_accuracy_interval
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
    write_csv,
)
from room_for_doubt.utils.paths import config_directory


def source_data(cfg):
    if not cfg.source_run:
        raise ValueError(
            "source_run must identify a completed base with real snapshots"
        )
    source = Path(to_absolute_path(cfg.source_run))
    config = OmegaConf.to_container(
        OmegaConf.load(source / "config.yaml"), resolve=True
    )
    cls = {
        "cifar10n": CIFAR10,
        "cifar100n": CIFAR100,
        "animal10n": Animal10N,
        "miniwebvision": MiniWebVision,
    }[config["dataset"]["name"]]
    root = to_absolute_path(cfg.data_dir)
    raw = cls(root, train=True, download=True)
    verify_identity(raw, dict(config, data_dir=root))
    noisy = load_observed_labels(
        root, config["dataset"]["name"], config["dataset"]["noisy_split"]
    )
    bundle = load_completed_run(source, noisy)
    return bundle, raw, noisy, cls


def mean_teachers(bundle, snapshots, images, ids, cfg, device, root, split):
    total = np.zeros(
        (len(ids), bundle["config"]["dataset"]["num_classes"]), dtype=np.float64
    )
    model = restore_base(bundle, device)
    for snapshot in snapshots:
        saved = torch.load(snapshot["path"], map_location="cpu", weights_only=False)
        model.load_state_dict(saved["state_dict"], strict=True)
        del saved
        _, p = extract(
            model,
            images,
            ids,
            bundle["config"]["image_size"],
            cfg.image_batch_size,
            device,
            root,
            f"{split}_epoch{snapshot['epoch']:03d}",
            False,
        )
        total += p
    result = (total / len(snapshots)).astype(np.float32)
    result /= result.sum(1, keepdims=True)
    return result


def prepare(cfg, directory, device):
    bundle, raw, noisy, _ = source_data(cfg)
    if not bundle["config"].get("defer_test_evaluation", False):
        raise ValueError(
            "The new pilot requires a base with deferred clean-test evaluation"
        )
    snapshots = verify_snapshots(
        bundle["source"], bundle["config"], list(cfg.teacher_epochs)
    )
    root = Path(cfg.cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    scientific = dict(
        base_digest=bundle["checkpoint_digest"],
        snapshots=[{k: s[k] for k in ("epoch", "sha256")} for s in snapshots],
        image_size=bundle["config"]["image_size"],
        acquisition="canonical eval FP32; end-of-epoch teacher probabilities",
        split_seed=bundle["config"]["split_seed"],
        dataset=bundle["config"]["dataset"],
    )
    protocol = root / "preparation_contract.json"
    if protocol.exists() and json.loads(protocol.read_text()) != scientific:
        raise ValueError("Existing temporal cache belongs to another source/protocol")
    protocol.write_text(json.dumps(scientific, indent=2))
    if (root / "manifest.json").exists():
        manifest, _ = load_evidence(root)
        if any(manifest[k] != value for k, value in scientific.items()):
            raise ValueError("Completed temporal cache contract differs")
        return dict(
            status="cache_already_verified", base_digest=bundle["checkpoint_digest"]
        )
    data = bundle["data"]
    train_ids, val_ids = data["base_train_ids"], data["base_val_ids"]
    model = restore_base(bundle, device)
    for split, ids in (("train", train_ids), ("val", val_ids)):
        extract(
            model,
            raw.data,
            ids,
            bundle["config"]["image_size"],
            cfg.image_batch_size,
            device,
            root,
            split,
            True,
        )
        np.save(root / f"{split}_ids.npy", ids)
    del model
    for split, ids, name in (
        ("train", train_ids, "teacher_probs"),
        ("val", val_ids, "val_teacher_probs"),
    ):
        q = mean_teachers(bundle, snapshots, raw.data, ids, cfg, device, root, split)
        np.save(root / f"{name}.npy", q)
    np.save(root / "val_noisy_labels.npy", noisy[val_ids])
    mapping = {int(sid): row for row, sid in enumerate(train_ids)}
    for name in ("train", "val"):
        np.save(
            root / f"meta_{name}_rows.npy",
            np.array(
                [mapping[int(s)] for s in data[f"meta_{name}_ids"]], dtype=np.int64
            ),
        )
    manifest = seal_evidence(
        root,
        dict(
            **scientific,
            classes=bundle["config"]["dataset"]["num_classes"],
            feature_width=bundle["checkpoint"]["state_dict"]["fc.weight"].shape[1],
            class_axis=list(range(bundle["config"]["dataset"]["num_classes"])),
        ),
    )
    out = Path(directory) / "prepare"
    out.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    config["dataset"] = manifest["dataset"]
    OmegaConf.save(OmegaConf.create(config), out / "config.yaml", resolve=True)
    metrics = dict(
        status="complete",
        train_images=len(train_ids),
        noisy_val_images=len(val_ids),
        snapshots=len(snapshots),
        classes=manifest["classes"],
        test_accessed=False,
    )
    save_metrics(out, metrics)
    save_run_metadata(
        out,
        cfg.run_id,
        config,
        base_digest=bundle["checkpoint_digest"],
        cache_manifest_sha256=file_digest(root / "manifest.json"),
        device=str(device),
    )
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        notes="Canonical all-class evidence from real retained checkpoints. No clean training labels or test data accessed. No accuracy gain measured.",
    )
    return metrics


def final_evaluate(cfg, directory, device):
    out = Path(directory) / "final_evaluation"
    head, policy = load_frozen(directory, cfg.cache_dir, device)
    manifest, _ = load_evidence(cfg.cache_dir)
    bundle, _, _, cls = source_data(cfg)
    if bundle["checkpoint_digest"] != policy["base_digest"]:
        raise ValueError("Evaluation base differs from frozen student")
    snapshots = verify_snapshots(
        bundle["source"], bundle["config"], policy["teacher_epochs"]
    )
    if [{k: s[k] for k in ("epoch", "sha256")} for s in snapshots] != manifest[
        "snapshots"
    ]:
        raise ValueError("Evaluation teachers differ from the sealed training evidence")
    evaluation_contract = dict(
        base_digest=policy["base_digest"],
        snapshots=manifest["snapshots"],
        cache_manifest_sha256=policy["cache_manifest_sha256"],
        policy_sha256=file_digest(
            Path(directory) / "artifacts/policy_before_test.json"
        ),
    )
    contract_path = out / "evaluation_contract.json"
    if out.exists():
        if (out / "metrics.json").exists():
            raise ValueError(
                "Final evaluation export already exists; preserve the original"
            )
        if (
            not contract_path.exists()
            or json.loads(contract_path.read_text()) != evaluation_contract
        ):
            raise ValueError("Incomplete evaluation belongs to another frozen contract")
    out.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(evaluation_contract, indent=2))
    # This is the first and only stage that instantiates clean final-evaluation data.
    raw = cls(to_absolute_path(cfg.data_dir), train=False, download=True)
    ids = np.arange(len(raw.data), dtype=np.int64)
    labels = np.asarray(raw.targets, dtype=np.int64)
    f, p = extract(
        restore_base(bundle, device),
        raw.data,
        ids,
        bundle["config"]["image_size"],
        cfg.image_batch_size,
        device,
        out,
        "test",
        True,
    )
    h = residuals(head, f, cfg.batch_size, device)
    corrected = corrected_probabilities(p, h, policy["policy"]["alpha"])
    teacher = mean_teachers(bundle, snapshots, raw.data, ids, cfg, device, out, "test")
    teacher_h = np.log(np.clip(teacher, 1e-12, 1)) - np.log(np.clip(p, 1e-12, 1))
    mixture = corrected_probabilities(p, teacher_h, policy["teacher_policy"]["alpha"])
    rows = evaluate_probabilities(labels, p, corrected)
    teacher_rows = evaluate_probabilities(labels, p, teacher)
    rows[f"snapshot_mean_{len(snapshots)}passes"] = teacher_rows["temporal_residual"]
    rows["teacher_interpolation_extra_passes"] = evaluate_probabilities(
        labels, p, mixture
    )["temporal_residual"]
    np.savez_compressed(
        out / "predictions.npz",
        ids=ids,
        labels=labels,
        base=p,
        corrected=corrected,
        snapshot_mean=teacher,
        teacher_interpolation=mixture,
    )
    metrics = dict(
        status="complete",
        methods=rows,
        paired_vs_identity=paired_accuracy_interval(
            labels, p.argmax(1), corrected.argmax(1)
        ),
        independent_bases=1,
        exploratory=True,
        policy_selected_before_test=True,
    )
    config = OmegaConf.to_container(cfg, resolve=True)
    config["dataset"] = bundle["config"]["dataset"]
    OmegaConf.save(OmegaConf.create(config), out / "config.yaml", resolve=True)
    info = dict(
        base_digest=policy["base_digest"],
        policy_sha256=file_digest(
            Path(directory) / "artifacts/policy_before_test.json"
        ),
        checkpoint_path=str(Path(directory) / "checkpoints/best.pt"),
        device=str(device),
    )
    save_run_metadata(out, cfg.run_id, config, **info)
    (out / "tables").mkdir(exist_ok=True)
    write_csv(
        out / "tables/methods.csv",
        [
            dict(
                method=name,
                **{
                    k: None if v is None else float(f"{v:.4g}")
                    for k, v in values.items()
                },
            )
            for name, values in rows.items()
        ],
    )
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        info,
        "One exploratory base. Conditional paired example interval, not between-base uncertainty. "
        f"Snapshot mean uses {len(snapshots)} backbone passes; teacher interpolation may require one additional selected-base pass. "
        "Student and identity each use one pass. No temporal-attribution or novelty claim.",
    )
    with (out / "report.md").open("a") as handle:
        handle.write(
            "\n## Method comparison\n\n| Method | Accuracy | NLL | ECE | Own-error AUROC |\n|---|---:|---:|---:|---:|\n"
        )
        for name, values in rows.items():
            auroc = values["own_error_auroc"]
            handle.write(
                f"| {name} | {values['acc']:.4g} | {values['nll']:.4g} | {values['ece']:.4g} | {'undefined' if auroc is None else f'{auroc:.4g}'} |\n"
            )
    # Commit completion only after all inspectable outputs have been written.
    save_metrics(out, metrics)
    return metrics


@hydra.main(
    config_path=str(config_directory()),
    config_name="temporal_residual",
    version_base=None,
)
def main(cfg):
    if cfg.stage not in ("prepare", "fit", "evaluate"):
        raise ValueError("stage must be prepare, fit or evaluate")
    for key in ("cache_dir", "data_dir", "output_dir"):
        cfg[key] = to_absolute_path(cfg[key])
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )
    if cfg.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "Enable CUDA or explicitly set require_cuda=false for local tests"
        )
    if cfg.stage == "fit":
        manifest, _ = load_evidence(cfg.cache_dir)
        cfg.dataset = manifest["dataset"]
    action = {"prepare": prepare, "fit": fit, "evaluate": final_evaluate}[cfg.stage]
    print(
        json.dumps(action(cfg, HydraConfig.get().runtime.output_dir, device), indent=2)
    )


if __name__ == "__main__":
    main()
