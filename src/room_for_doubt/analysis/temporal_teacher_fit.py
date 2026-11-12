"""Fit and seal direct controls against the frozen S4 Early-5 student."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import torch
from omegaconf import OmegaConf

from room_for_doubt.analysis.temporal_distillation import prepare, source_data
from room_for_doubt.data.confirmation import CLASSES, verify_frozen, write_json
from room_for_doubt.data.kaggle_artifacts import restore_base
from room_for_doubt.data.temporal_evidence import load_evidence
from room_for_doubt.evaluation.diagnostic_metrics import probability_metrics
from room_for_doubt.inference.static_teacher import STATIC_VIEWS, static_teacher_probabilities
from room_for_doubt.inference.temporal_evidence import extract
from room_for_doubt.reporting.artifacts import write_csv
from room_for_doubt.training.posthoc_controls import fit_head
from room_for_doubt.training.temporal_snapshots import file_digest, verify_snapshots


BRANCHES = ("static5_direct", "selected1_direct")
FROZEN_FILES = (
    "config.yaml",
    "metrics.json",
    "run_metadata.json",
    "report.md",
    "selection.csv",
    *(
        f"{branch}/{name}"
        for branch in BRANCHES
        for name in ("config.yaml", "metrics.json", "checkpoints/best.pt",
                     "artifacts/policy_before_test.json")
    ),
)


def fit_controls(source, out, scratch, data, seed, s4_hash, protocol_hash, device):
    source, out, scratch = map(Path, (source, out, scratch))
    verify_frozen(source, seed, s4_hash)
    config = dict(
        seed=seed, source_frozen_sha256=file_digest(source / "frozen.json"),
        protocol_sha256=protocol_hash, s4_protocol_sha256=s4_hash,
        data_dir=str(data), image_batch_size=64, views=list(STATIC_VIEWS),
        device=str(device), test_accessed=False,
    )
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.yaml").exists():
        previous = OmegaConf.to_container(OmegaConf.load(out / "config.yaml"))
        if previous != config:
            raise ValueError("Existing control configuration differs")
    if (out / "frozen.json").exists():
        frozen = json.loads((out / "frozen.json").read_text())
        expected = dict(config=config, files={name: file_digest(out / name) for name in FROZEN_FILES})
        if frozen != expected:
            raise ValueError("Frozen control artifact changed")
        return json.loads((out / "metrics.json").read_text())
    OmegaConf.save(OmegaConf.create(config), out / "config.yaml")

    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=scratch) as directory:
        work = Path(directory)
        source_cache = source / "evidence"
        sealed_cache = json.loads((source_cache / "manifest.json").read_text())
        cache = source_cache
        started = monotonic()
        cfg = OmegaConf.create(dict(
            run_id=f"teacher_control_seed{seed}", source_run=str(source / "base"),
            cache_dir=str(work / "evidence"), data_dir=str(data),
            teacher_epochs=[1, 2, 3, 4, 5], image_batch_size=64,
        ))
        bundle, raw, noisy, _ = source_data(cfg)
        if tuple(raw.classes) != CLASSES:
            raise ValueError("Unexpected CIFAR-10 class order")
        if not all((cache / f"{name}.npy").is_file() for name in sealed_cache["arrays"]):
            prepare(cfg, work, device)
            cache = work / "evidence"
        manifest, arrays = load_evidence(cache)
        # Fallback extraction may differ at FP32 roundoff; source and partitions must match.
        for key in ("base_digest", "snapshots", "dataset", "split_seed", "class_axis",
                    "image_size", "classes", "feature_width"):
            if manifest[key] != sealed_cache[key]:
                raise ValueError(f"Unexpected S4 evidence field: {key}")
        train_ids, val_ids = arrays["train_ids"], arrays["val_ids"]
        for ids, split in ((train_ids, "train"), (val_ids, "val")):
            np.testing.assert_array_equal(ids, bundle["data"][f"base_{split}_ids"])
        np.testing.assert_array_equal(arrays["val_noisy_labels"], noisy[val_ids])
        if arrays["train_features"].shape[1] != 2048 or arrays["train_probs"].shape[1] != 10:
            raise ValueError("Unexpected S4 features or probabilities")

        model = restore_base(bundle, device)
        image_size = bundle["config"]["image_size"]
        batch = cfg.image_batch_size
        snapshots = verify_snapshots(bundle["source"], bundle["config"], [1, 2, 3, 4, 5])
        rows = []
        for snapshot in snapshots:
            epoch = snapshot["epoch"]
            model.load_state_dict(
                torch.load(snapshot["path"], map_location="cpu", weights_only=False)["state_dict"],
                strict=True,
            )
            _, val = extract(model, raw.data, val_ids, image_size, batch, device,
                             work, f"val_epoch{epoch}", False)
            metrics = probability_metrics(noisy[val_ids], val)
            rows.append(dict(epoch=epoch, acc=metrics["acc"], nll=metrics["nll"]))
        selected = max(rows, key=lambda row: (row["acc"], -row["nll"], -row["epoch"]))
        write_csv(out / "selection.csv", rows)
        snapshot = next(item for item in snapshots if item["epoch"] == selected["epoch"])
        model.load_state_dict(
            torch.load(snapshot["path"], map_location="cpu", weights_only=False)["state_dict"],
            strict=True,
        )
        _, train_single = extract(model, raw.data, train_ids, image_size, batch, device,
                                  work, "train_single", False)
        val_single = np.load(work / f"val_epoch{selected['epoch']}_probs.npy", allow_pickle=False)
        static = [
            static_teacher_probabilities(model, raw.data, ids, image_size, batch, device, work)
            for ids in (train_ids, val_ids)
        ]
        del model
        extraction_seconds = monotonic() - started
        policies = {}
        for branch, targets in zip(BRANCHES, (static, (train_single, val_single))):
            digests = {}
            for split, target in zip(("train", "val"), targets):
                path = work / f"{branch}_{split}.npy"
                np.save(path, target, allow_pickle=False)
                digests[split] = file_digest(path)
            contract = dict(
                protocol="temporal_teacher_v1", protocol_sha256=protocol_hash,
                source_frozen_sha256=config["source_frozen_sha256"],
                recipient=seed, branch=branch, parametrization="direct",
                classes=10, hidden_dim=256, head_seed=42, dataset=manifest["dataset"],
                evidence_sha256=file_digest(cache / "manifest.json"),
                base_digest=manifest["base_digest"], target_sha256=digests,
                teacher_epoch=selected["epoch"], teacher_sha256=snapshot["sha256"],
            )
            policies[branch] = fit_head(out / branch, arrays, *targets, contract, device)["policy"]
    metrics = dict(status="frozen", selected_epoch=selected["epoch"],
                   target_preparation_seconds=extraction_seconds, policies=policies,
                   test_accessed=False)
    write_json(out / "metrics.json", metrics)
    write_json(out / "run_metadata.json", config)
    (out / "report.md").write_text(
        f"# Direct distillation controls, seed {seed}\n\n"
        f"Selected teacher epoch {selected['epoch']} on noisy validation.\n\n"
        "See selection.csv and both branch reports. No external evaluation performed.\n"
    )
    write_json(out / "frozen.json",
               dict(config=config, files={name: file_digest(out / name) for name in FROZEN_FILES}))
    return metrics
