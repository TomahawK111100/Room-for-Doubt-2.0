"""One unattended diagnostic battery on completed base and downstream runs."""

import json
import shutil
import time
import zipfile
from pathlib import Path
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torchvision.datasets import CIFAR10, CIFAR100
from room_for_doubt.data.cifar_noisy import load_observed_labels
from room_for_doubt.data.kaggle_artifacts import load_completed_run
from room_for_doubt.evaluation.diagnostic_metrics import class_average
from room_for_doubt.evaluation.diagnostic_study import (
    reconstruction,
    freeze_diagnostic_policies,
    evaluate_diagnostic_policies,
    noise_diagnostics,
)
from room_for_doubt.inference.diagnostic_cache import (
    best_train_predictions,
    numerical_audit,
)
from room_for_doubt.training.diagnostic_controls import train_control
from room_for_doubt.training.meta_retraining import choose_device
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
    write_csv,
)
from room_for_doubt.utils.paths import config_directory


def check_post(bundle, post, cfg):
    post = Path(post)
    post_cfg = OmegaConf.load(post / "config.yaml")
    source = json.loads((post / "artifacts/source.json").read_text())
    original = json.loads(
        (post / "artifacts/all_policies_before_test.json").read_text()
    )
    if (
        source["base_digest"] != bundle["checkpoint_digest"]
        or original["base_digest"] != bundle["checkpoint_digest"]
    ):
        raise ValueError("Base and downstream run do not belong together")
    if json.loads((post / "metrics.json").read_text()).get("status") != "complete":
        raise ValueError("Complete the previous downstream run first")
    for key in (
        "epochs",
        "batch_size",
        "learning_rate",
        "split_seed",
        "seeds",
        "temperatures",
        "inference_seed",
    ):
        if (
            OmegaConf.to_container(post_cfg, resolve=True)[key]
            != OmegaConf.to_container(cfg, resolve=True)[key]
        ):
            raise ValueError("Matched diagnostic setting must equal source: " + key)
    if list(post_cfg.modes) != ["mean", "full"]:
        raise ValueError("Require completed mean/full source heads")
    counts = list(cfg.view_counts)
    if (
        not counts
        or counts != sorted(set(counts))
        or counts[0] < 1
        or counts[-1] > post_cfg.views
    ):
        raise ValueError("View counts must be ordered unique prefixes of cached views")
    if not 1 <= cfg.early_epochs <= bundle["data"]["targets"].shape[1]:
        raise ValueError("Invalid early-history budget")
    for split, expected in (("val", bundle["data"]["base_val_ids"]), ("test", None)):
        protocol = json.loads((post / "cache" / f"{split}_complete.json").read_text())
        ids = np.load(post / "cache" / f"{split}_ids.npy")
        if (
            protocol["base_digest"] != bundle["checkpoint_digest"]
            or protocol["views"] != post_cfg.views
            or protocol["seed"] != cfg.inference_seed
            or protocol.get("image_size") != bundle["config"]["image_size"]
        ):
            raise ValueError("Source view-cache protocol mismatch")
        if not np.array_equal(ids, protocol["ids"]) or (
            expected is not None and not np.array_equal(ids, expected)
        ):
            raise ValueError("Source cache IDs differ")
        p = np.load(post / "cache" / f"{split}_probs.npy", mmap_mode="r")
        f = np.load(post / "cache" / f"{split}_features.npy", mmap_mode="r")
        if p.shape != (
            len(ids),
            post_cfg.views,
            cfg.dataset.num_classes,
        ) or f.shape != (len(ids), post_cfg.views, 512):
            raise ValueError("Incomplete view cache")
    with np.load(post / "features/features.npz") as f:
        if not np.array_equal(f["sample_ids"], bundle["data"]["sample_ids"]):
            raise ValueError("Source feature IDs differ")
        if (
            f["dataset_name"].item() != cfg.dataset.name
            or f["preprocessing"].item() != "imagenet_pil_bilinear_224"
        ):
            raise ValueError("Source feature dataset/preprocessing differs")
        features = f["features"]
    if (
        features.shape != (len(bundle["data"]["sample_ids"]), 512)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Invalid source frozen features")
    return post_cfg, features


def export_summary(out, cfg, metrics, metadata):
    out = Path(out)
    config = OmegaConf.to_container(cfg, resolve=True)
    save_metrics(out, metrics)
    save_run_metadata(out, cfg.run_id, config, **metadata)
    notes = (
        "Fixed exploratory diagnostic battery after previous CIFAR test inspection; not an untouched confirmatory experiment. "
        "No base retraining. All nine new scalar heads use noisy observed labels, shared frozen features, the original meta partitions and minimum meta-validation MSE. "
        "All temperatures for every fixed view budget/method are frozen on noisy validation before diagnostic test metrics. "
        "Every comparison is reported, with no test-selected winner. Paired intervals are conditional on this single base model, nominal and uncorrected for multiple comparisons. "
        "Clean training labels are read only after policy freezing for final train-noise diagnostic evaluation; those examples were seen by the base, but detector_eval was held out of meta fitting and checkpoint selection. "
        "Best-checkpoint original-image targets and augmented train-time history targets differ in acquisition as well as temporal coverage; this is a disclosed control limitation. "
        "No AURC is reported. Failure or time-budget pause preserves partial outputs."
    )
    generate_report(out, cfg.run_id, config, metrics, metadata, notes)
    lines = [
        f"Status: {metrics['status']}",
        f"Completed stage: {metrics.get('completed_stage', 'initialization')}",
    ]
    if metrics.get("test_rows"):
        count = max(cfg.view_counts)
        lines.append(
            f"Fixed {count}-view comparison (single base; meta seeds are separate rows):"
        )
        for r in metrics["test_rows"]:
            if r["views"] == count:
                lines.append(
                    f"{r['method']}: acc={100 * r['acc']:.2f}%, delta_base={r['gain_pp_vs_base']:+.2f}pp [{r['ci_low_pp_vs_base']:+.2f},{r['ci_high_pp_vs_base']:+.2f}], error_AUROC={r['base_error_auroc']}"
                )
        with (out / "report.md").open("a") as handle:
            handle.write(
                "\n## Fixed maximum-view comparison\n\n| Method | Acc, % | Gain vs base, pp | Nominal 95% CI, pp |\n|---|---:|---:|---|\n"
            )
            for r in metrics["test_rows"]:
                if r["views"] == count:
                    handle.write(
                        f"| {r['method']} | {100 * r['acc']:.2f} | {r['gain_pp_vs_base']:+.2f} | [{r['ci_low_pp_vs_base']:+.2f}, {r['ci_high_pp_vs_base']:+.2f}] |\n"
                    )
    (out / "morning_summary.txt").write_text("\n".join(lines) + "\n")
    with zipfile.ZipFile(
        out / "summary_bundle.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in (
            "config.yaml",
            "metrics.json",
            "run_metadata.json",
            "report.md",
            "morning_summary.txt",
            "artifacts/diagnostic_policies_before_test.json",
            "artifacts/failure.json",
        ):
            path = out / name
            if path.exists():
                archive.write(path, name)
        for path in sorted((out / "tables").glob("*.csv")):
            archive.write(path, str(path.relative_to(out)))
    print(
        "\n" + "\n".join(lines) + f"\nSummary archive: {out / 'summary_bundle.zip'}",
        flush=True,
    )


def run_diagnostics(cfg, out):
    started = time.monotonic()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if not cfg.source_run or not cfg.post_run:
        raise ValueError("Both source_run and post_run are required")
    if list(cfg.control_modes) != ["best_margin", "last_margin", "early_mean"]:
        raise ValueError("Fixed diagnostic battery has exactly three control targets")
    if (
        cfg.image_batch_size < 1
        or cfg.max_hours <= 0
        or cfg.optimizer != "adam"
        or cfg.meta_model != "conditioned"
    ):
        raise ValueError(
            "Positive budgets and matched Adam/conditioned settings required"
        )
    source, post, out = (
        Path(cfg.source_run).resolve(),
        Path(cfg.post_run).resolve(),
        Path(out).resolve(),
    )
    if out in (source, post):
        raise ValueError("Diagnostic output must be separate from input runs")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("artifacts", "tables", "heads"):
        (out / name).mkdir(exist_ok=True)
    if shutil.disk_usage(out).free < cfg.min_free_gb * 1024**3:
        raise RuntimeError("Insufficient disk space for diagnostic controls")
    source_cfg = OmegaConf.load(source / "config.yaml")
    observed = load_observed_labels(
        cfg.data_dir, source_cfg.dataset.name, source_cfg.dataset.noisy_split
    )
    bundle = load_completed_run(source, observed)
    cfg.dataset = bundle["config"]["dataset"]
    cfg.base_digest = bundle["checkpoint_digest"]
    post_cfg, features = check_post(bundle, post, cfg)
    if not np.array_equal(
        np.load(post / "cache/val_labels.npy"), observed[bundle["data"]["base_val_ids"]]
    ):
        raise ValueError("Validation cache does not contain source observed labels")
    cfg_path = out / "config.yaml"
    current = OmegaConf.to_container(cfg, resolve=True)
    if cfg_path.exists():
        previous = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
        # Paths/runtime limits may change after attaching an earlier version; scientific fields cannot.
        runtime = {
            "source_run",
            "post_run",
            "data_dir",
            "output_dir",
            "device",
            "max_hours",
            "min_free_gb",
            "image_batch_size",
        }
        if {k: v for k, v in previous.items() if k not in runtime} != {
            k: v for k, v in current.items() if k not in runtime
        }:
            raise ValueError("Existing diagnostic scientific configuration differs")
    OmegaConf.save(cfg, cfg_path, resolve=True)
    device = choose_device(cfg.device)
    cls = {"cifar10n": CIFAR10, "cifar100n": CIFAR100}[cfg.dataset.name]
    metadata = dict(
        seed=list(cfg.seeds),
        device=str(device),
        base_digest=cfg.base_digest,
        checkpoint_path=str(source / "checkpoints/best.pt"),
        best_epoch=bundle["checkpoint"]["epoch"],
        best_val_loss=bundle["checkpoint"]["val_loss"],
        code_snapshot="diagnostic_20261016",
        source_post_run=str(post),
    )
    metrics = dict(
        status="running",
        completed_stage="source_verified",
        base_retrained=False,
        test_evaluated=False,
    )

    def checkpoint(stage):
        metrics["completed_stage"] = stage
        metrics["elapsed_seconds"] = time.monotonic() - started
        save_metrics(out, metrics)
        if metrics["elapsed_seconds"] > cfg.max_hours * 3600:
            metrics["status"] = "paused_time_budget"
            export_summary(out, cfg, metrics, metadata)
            return False
        return True

    save_metrics(out, metrics)
    try:
        raw = cls(cfg.data_dir, train=True, download=False)
        p, best_margin = best_train_predictions(bundle, raw, out, cfg, device)
        if not checkpoint("static_targets_prepared"):
            return metrics
        for mode in cfg.control_modes:
            for seed in cfg.seeds:
                train_control(
                    cfg, bundle, post, out, features, best_margin, mode, seed, device
                )
                if not checkpoint(f"control_{mode}_{seed}"):
                    return metrics
        reconstruction(cfg, bundle, post_cfg, post, out, features, best_margin, device)
        if not checkpoint("reconstruction_complete"):
            return metrics
        data = bundle["data"]
        lookup = {int(s): i for i, s in enumerate(data["sample_ids"])}
        fit = np.array([lookup[int(s)] for s in data["meta_train_ids"]])
        class_values = class_average(
            data["targets"].mean(1, keepdims=True),
            data["noisy_labels"],
            fit,
            cfg.dataset.num_classes,
        )[:, 0]
        frozen = freeze_diagnostic_policies(
            cfg, bundle, post_cfg, post, out, class_values, device
        )
        if not checkpoint("all_policies_frozen"):
            return metrics
        # No test correctness or clean training labels is accessed before this point.
        test = cls(cfg.data_dir, train=False, download=False)
        cached_ids = np.load(post / "cache/test_ids.npy")
        if not np.array_equal(
            np.asarray(test.targets)[cached_ids],
            np.load(post / "cache/test_labels.npy"),
        ):
            raise ValueError("Cached test labels differ from dataset ground truth")
        metrics["test_rows"] = evaluate_diagnostic_policies(
            cfg, bundle, post_cfg, post, out, class_values, device, frozen
        )
        metrics["test_evaluated"] = True
        if not checkpoint("test_comparison_complete"):
            return metrics
        noise_diagnostics(cfg, bundle, raw.targets, p, out)
        if not checkpoint("train_noise_diagnostic_complete"):
            return metrics
        metrics["numerical_audit"] = numerical_audit(
            bundle, post, out, test, cfg, device
        )
        with (source / "history.csv").open() as handle:
            import csv

            history = list(csv.DictReader(handle))
        write_csv(out / "tables/base_training_history.csv", history)
        metrics["status"] = "complete"
        metrics["completed_stage"] = "all_diagnostics_complete"
        metrics["elapsed_seconds"] = time.monotonic() - started
        metrics["controls_trained_or_reused"] = len(cfg.control_modes) * len(cfg.seeds)
        export_summary(out, cfg, metrics, metadata)
        return metrics
    except Exception as exc:
        metrics["status"] = "failed"
        metrics["elapsed_seconds"] = time.monotonic() - started
        (out / "artifacts/failure.json").write_text(
            json.dumps(
                dict(
                    error_type=type(exc).__name__,
                    error=str(exc),
                    completed_stage=metrics["completed_stage"],
                ),
                indent=2,
            )
        )
        export_summary(out, cfg, metrics, metadata)
        raise


@hydra.main(
    config_path=str(config_directory()),
    config_name="diagnose_kaggle",
    version_base=None,
)
def main(cfg):
    run_diagnostics(cfg, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
