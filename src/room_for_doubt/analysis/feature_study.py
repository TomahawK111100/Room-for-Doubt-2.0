"""Frozen ResNet18 versus selected-base features with matched scalar heads."""

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
from room_for_doubt.analysis.kaggle_diagnostics import check_post
from room_for_doubt.data.cifar_noisy import load_observed_labels
from room_for_doubt.data.animal10n import Animal10N, verify_identity
from room_for_doubt.data.kaggle_artifacts import load_completed_run
from room_for_doubt.inference.classifier_features import (
    cache_classifier_features,
    fit_projection,
    project_features,
)
from room_for_doubt.inference.post_kaggle_cache import prepare_features, split_cache
from room_for_doubt.training.feature_heads import train_feature_head
from room_for_doubt.training.meta_retraining import choose_device
from room_for_doubt.evaluation.feature_study import (
    head_specs,
    freeze_policies,
    evaluate_policies,
    reconstruction,
    noise_and_epoch_profile,
)
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
    write_csv,
)
from room_for_doubt.utils.paths import config_directory


def export(out, cfg, metrics, metadata, notes=None):
    out = Path(out)
    config = OmegaConf.to_container(cfg, resolve=True)
    save_metrics(out, metrics)
    save_run_metadata(out, cfg.run_id, config, **metadata)
    notes = notes or (
        f"Fixed exploratory {len(cfg.encoders)}-encoder x {len(cfg.target_modes)}-target x {len(cfg.seeds)}-meta-seed study. "
        "Both encoders use the same PCA output dimension, scalar MLP, optimizer, fitting partitions and epoch budget. "
        "PCA is fitted only on meta-training images without labels, without whitening; native dimensions and retained variance are reported. "
        "ResNet18 uses external ImageNet initialization; ResNet50 uses the selected, noisy-label-finetuned base. "
        "This distinguishes representation pipelines, not solely architecture. "
        "All temperatures AND view budgets are selected on noisy validation; ties prefer fewer views and uniform weights. "
        "Clean test is accessed after the complete policy marker. "
        "Train-noise evaluation uses base-seen images held out from meta fitting only where clean training labels exist; Animal-10N has none and skips it. "
        "Confidence intervals are nominal, conditional on one base, and uncorrected for multiple comparisons. "
        "No base retraining or test-based method selection. Static original-image targets and train-time augmented history differ in acquisition. "
        "Matched heads share parameter count; these new PCA heads are not identical to previous unprojected ResNet18 heads."
    )
    generate_report(out, cfg.run_id, config, metrics, metadata, notes)
    lines = [
        f"Status: {metrics['status']}",
        f"Completed stage: {metrics.get('completed_stage')}",
    ]
    if metrics.get("aggregate"):
        lines.append("Joint noisy-validation-selected view budget and temperature:")
        with (out / "report.md").open("a") as f:
            f.write(
                "\n## Validation-selected comparison\n\n| Method | Accuracy, % | Error AUROC | Views |\n|---|---:|---:|---|\n"
            )
            for row in metrics["aggregate"]:
                line = f"{row['method']}: acc={100 * row['acc']:.2f}%, sd={100 * row['acc_sd']:.3f}pp, error_AUROC={row['error_auroc']:.4f}, views={row['selected_views']}"
                lines.append(line)
                f.write(
                    f"| {row['method']} | {100 * row['acc']:.2f} | {row['error_auroc']:.4f} | {row['selected_views']} |\n"
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
            "artifacts/policies_before_test.json",
            "artifacts/failure.json",
        ):
            path = out / name
            if path.exists():
                archive.write(path, name)
        for path in (out / "tables").glob("*.csv"):
            archive.write(path, str(path.relative_to(out)))
    print(
        "\n" + "\n".join(lines) + f"\nDownload: {out / 'summary_bundle.zip'}",
        flush=True,
    )


def run_study(cfg, out):
    started = time.monotonic()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    targets = (
        ["early_mean", "history_mean", "best_margin"]
        if cfg.scope == "standard"
        else [
            "early_mean_2",
            "early_mean",
            "early_mean_10",
            "early_mean_20",
            "history_mean",
            "late_mean_5",
            "best_margin",
        ]
    )
    if (
        cfg.scope not in {"standard", "extended"}
        or list(cfg.encoders) != ["resnet18", "base_resnet50"]
        or list(cfg.target_modes) != targets
    ):
        raise ValueError("Use the fixed encoder-by-target study")
    if (
        not cfg.source_run
        or cfg.feature_dim < 1
        or cfg.max_hours <= 0
        or cfg.image_batch_size < 1
    ):
        raise ValueError("Source runs and positive budgets are required")
    if cfg.optimizer != "adam" or cfg.meta_model != "conditioned":
        raise ValueError("Matched feature heads use Adam and conditioned regression")
    source, out = Path(cfg.source_run).resolve(), Path(out).resolve()
    supplied_post = bool(cfg.post_run)
    post = Path(cfg.post_run).resolve() if supplied_post else out / "shared"
    if out in (source, post):
        raise ValueError("Keep experiment output separate from source inputs")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("artifacts", "tables", "projected", "heads"):
        (out / name).mkdir(exist_ok=True)
    if shutil.disk_usage(out).free < cfg.min_free_gb * 1024**3:
        raise RuntimeError("Need working storage for native and projected features")
    source_cfg = OmegaConf.load(source / "config.yaml")
    noisy = load_observed_labels(
        cfg.data_dir, source_cfg.dataset.name, source_cfg.dataset.noisy_split
    )
    bundle = load_completed_run(source, noisy)
    cfg.dataset, cfg.base_digest = (
        bundle["config"]["dataset"],
        bundle["checkpoint_digest"],
    )
    cfg.base_seed = bundle["config"]["seed"]
    counts = list(cfg.view_counts)
    if not counts or counts != sorted(set(counts)) or counts[0] < 1:
        raise ValueError("Use positive ordered view budgets")
    if cfg.split_seed != bundle["config"]["split_seed"]:
        raise ValueError("Meta partition seed must match source")
    cfg.views = max(counts)
    old_train_features = None
    if supplied_post:
        _, old_train_features = check_post(bundle, post, cfg)
    val_ids = bundle["data"]["base_val_ids"]
    if supplied_post and not np.array_equal(
        np.load(post / "cache/val_labels.npy"), noisy[val_ids]
    ):
        raise ValueError("Validation cache labels differ from observed annotations")
    config_path = out / "config.yaml"
    config = OmegaConf.to_container(cfg, resolve=True)
    if config_path.exists():
        previous = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
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
            k: v for k, v in config.items() if k not in runtime
        }:
            raise ValueError("Existing feature-study scientific configuration differs")
    OmegaConf.save(cfg, config_path, resolve=True)
    device = choose_device(cfg.device)
    cls = {"cifar10n": CIFAR10, "cifar100n": CIFAR100, "animal10n": Animal10N}[
        cfg.dataset.name
    ]
    meta = dict(
        base_seed=cfg.base_seed,
        seed=list(cfg.seeds),
        device=str(device),
        checkpoint_path=str(source / "checkpoints/best.pt"),
        best_epoch=bundle["checkpoint"]["epoch"],
        best_val_loss=bundle["checkpoint"]["val_loss"],
        source_post_run=str(post),
        code_snapshot="feature_study_20261016",
    )
    metrics = dict(
        status="running",
        completed_stage="source_verified",
        base_retrained=False,
        test_evaluated=False,
    )

    def checkpoint(stage):
        metrics["completed_stage"], metrics["elapsed_seconds"] = (
            stage,
            time.monotonic() - started,
        )
        save_metrics(out, metrics)
        if metrics["elapsed_seconds"] > cfg.max_hours * 3600:
            metrics["status"] = "paused_time_budget"
            export(out, cfg, metrics, meta)
            return False
        return True

    try:
        raw = cls(cfg.data_dir, train=True, download=not supplied_post)
        verify_identity(raw, dict(bundle["config"], data_dir=str(cfg.data_dir)))
        if not supplied_post:
            prepare_features(bundle, raw.data, post / "features", cfg, device)
            with np.load(post / "features/features.npz") as f:
                if not np.array_equal(f["sample_ids"], bundle["data"]["sample_ids"]):
                    raise ValueError("Prepared feature IDs differ")
                old_train_features = f["features"]
        ids = bundle["data"]["sample_ids"]
        native, train_probs = cache_classifier_features(
            bundle, raw.data, ids, "train", out, cfg, device
        )
        native = native[:, 0]
        p = train_probs[:, 0]
        observed = bundle["data"]["noisy_labels"]
        other = np.array(p)
        other[np.arange(len(p)), observed] = -1
        best_margin = p[np.arange(len(p)), observed] - other.max(1)
        np.savez_compressed(
            out / "artifacts/training_targets.npz",
            **bundle["data"],
            best_margin=best_margin,
        )
        lookup = {int(s): i for i, s in enumerate(ids)}
        fit = np.array([lookup[int(s)] for s in bundle["data"]["meta_train_ids"]])
        projections, train_features, projection_rows = {}, {}, []
        for encoder, features in (
            ("resnet18", old_train_features),
            ("base_resnet50", native),
        ):
            branch = out / "projected" / encoder
            branch.mkdir(exist_ok=True)
            projections[encoder] = fit_projection(
                features, fit, branch / "projection.npz", cfg.feature_dim, cfg.pca_seed
            )
            train_features[encoder] = project_features(
                features, projections[encoder], branch / "train.npy"
            )
            projection_rows.append(
                dict(
                    encoder=encoder,
                    native_dim=features.shape[-1],
                    output_dim=cfg.feature_dim,
                    fit_samples=len(fit),
                    retained_variance=float(
                        projections[encoder]["explained_variance_ratio"].sum()
                    ),
                )
            )
        write_csv(out / "tables/projections.csv", projection_rows)
        if not checkpoint("training_features_prepared"):
            return metrics
        for encoder, mode, seed in head_specs(cfg):
            model = train_feature_head(
                cfg,
                bundle,
                out,
                train_features[encoder],
                best_margin,
                encoder,
                mode,
                seed,
                device,
            )
            del model
            if not checkpoint(f"head_{encoder}_{mode}_{seed}"):
                return metrics
        reconstruction(cfg, bundle, out, train_features, best_margin, device)
        count = max(cfg.view_counts)
        if not supplied_post:
            split_cache(
                bundle,
                raw.data,
                val_ids,
                noisy,
                "val",
                post / "cache",
                post / "features",
                cfg,
                device,
            )
        val_probs = np.load(post / "cache/val_probs.npy", mmap_mode="r")[:, :count]
        val_native, _ = cache_classifier_features(
            bundle, raw.data, val_ids, "val", out, cfg, device, val_probs
        )
        val_features = {}
        for encoder, features in (
            (
                "resnet18",
                np.load(post / "cache/val_features.npy", mmap_mode="r")[:, :count],
            ),
            ("base_resnet50", val_native),
        ):
            val_features[encoder] = project_features(
                features, projections[encoder], out / "projected" / encoder / "val.npy"
            )
        frozen = freeze_policies(
            cfg, out, val_probs, noisy[val_ids], val_features, device
        )
        if not checkpoint("all_policies_frozen"):
            return metrics
        # The test dataset and clean training labels are unavailable to fitting/selection.
        test = cls(cfg.data_dir, train=False, download=not supplied_post)
        if not supplied_post:
            split_cache(
                bundle,
                test.data,
                np.arange(len(test.data)),
                np.asarray(test.targets),
                "test",
                post / "cache",
                post / "features",
                cfg,
                device,
            )
        test_ids = np.load(post / "cache/test_ids.npy")
        test_labels = np.asarray(test.targets)[test_ids]
        if not np.array_equal(test_labels, np.load(post / "cache/test_labels.npy")):
            raise ValueError("Test cache does not match official ground truth")
        test_probs = np.load(post / "cache/test_probs.npy", mmap_mode="r")[:, :count]
        test_native, _ = cache_classifier_features(
            bundle, test.data, test_ids, "test", out, cfg, device, test_probs
        )
        test_features = {}
        for encoder, features in (
            (
                "resnet18",
                np.load(post / "cache/test_features.npy", mmap_mode="r")[:, :count],
            ),
            ("base_resnet50", test_native),
        ):
            test_features[encoder] = project_features(
                features, projections[encoder], out / "projected" / encoder / "test.npy"
            )
        metrics["selected_results"], metrics["aggregate"] = evaluate_policies(
            cfg, out, test_probs, test_labels, test_features, device, frozen
        )
        metrics["test_evaluated"] = True
        if cfg.dataset.name != "animal10n":
            noise_and_epoch_profile(cfg, bundle, out, raw.targets)
        else:
            metrics["train_noise_diagnostic"] = "unavailable_no_clean_training_labels"
        metrics.update(
            status="complete",
            completed_stage="all_study_stages_complete",
            elapsed_seconds=time.monotonic() - started,
            heads_trained_or_reused=len(head_specs(cfg)),
        )
        export(out, cfg, metrics, meta)
        return metrics
    except Exception as exc:
        metrics.update(status="failed", elapsed_seconds=time.monotonic() - started)
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
        export(out, cfg, metrics, meta)
        raise


@hydra.main(
    config_path=str(config_directory()), config_name="feature_study", version_base=None
)
def main(cfg):
    run_study(cfg, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
