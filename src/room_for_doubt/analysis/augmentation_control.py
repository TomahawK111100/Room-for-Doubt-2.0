"""A bounded fixed-checkpoint augmentation control using completed study assets."""

import json
import shutil
import time
from pathlib import Path
import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torchvision.datasets import CIFAR10, CIFAR100
from room_for_doubt.analysis.feature_study import export
from room_for_doubt.data.cifar_noisy import load_observed_labels
from room_for_doubt.data.feature_artifacts import completed_study, shared_split
from room_for_doubt.data.kaggle_artifacts import load_completed_run, checkpoint_digest
from room_for_doubt.inference.augmented_margin import augmented_margin
from room_for_doubt.inference.classifier_features import fit_projection
from room_for_doubt.training.feature_heads import train_feature_head, load_feature_head
from room_for_doubt.training.meta_retraining import choose_device
from room_for_doubt.evaluation.feature_study import (
    freeze_policies,
    evaluate_policies,
    reconstruction,
)
from room_for_doubt.evaluation.paired_targets import target_contrasts
from room_for_doubt.utils.paths import config_directory


def run_control(launch, out):
    if not launch.study_run or not launch.source_run:
        raise ValueError("Require complete base and feature-study Outputs")
    study, source, out = [
        Path(p).resolve() for p in (launch.study_run, launch.source_run, out)
    ]
    post = Path(launch.post_run).resolve() if launch.post_run else study / "shared"
    for parent in (source, study, post):
        if out == parent or out in parent.parents or parent in out.parents:
            raise ValueError("Control output must be separate from all source inputs")
    previous, original = completed_study(study)
    if "base_resnet50" not in previous.encoders or not all(
        mode in previous.target_modes
        for mode in ("early_mean", "history_mean", "best_margin")
    ):
        raise ValueError("Source study lacks the declared matched controls")
    cfg = OmegaConf.merge(previous, launch)
    cfg.encoders = ["base_resnet50"]
    cfg.target_modes = [
        "early_mean",
        "history_mean",
        "best_margin",
        "best_augmented_mean",
    ]
    cfg.scope = "augmentation_control"
    cfg.feature_cache_root = str(study / "projected")
    cfg.post_run = str(post)
    cfg.static_target_digest = None
    if cfg.augmentation_views != cfg.early_epochs or cfg.max_hours <= 0:
        raise ValueError(
            "Static augmentation count must match the Early-K target; positive time budget required"
        )
    if (
        shutil.disk_usage(out.parent if out.parent.exists() else Path.cwd()).free
        < cfg.min_free_gb * 1024**3
    ):
        raise RuntimeError("Insufficient storage for the control")
    source_cfg = OmegaConf.load(source / "config.yaml")
    noisy = load_observed_labels(
        cfg.data_dir, source_cfg.dataset.name, source_cfg.dataset.noisy_split
    )
    bundle = load_completed_run(source, noisy)
    if (
        previous.base_digest != bundle["checkpoint_digest"]
        or OmegaConf.to_container(previous.dataset) != bundle["config"]["dataset"]
    ):
        raise ValueError("Base checkpoint or dataset differs from the source study")
    cfg.dataset = bundle["config"]["dataset"]
    cfg.base_digest = bundle["checkpoint_digest"]
    for name in ("artifacts", "tables", "heads"):
        (out / name).mkdir(parents=True, exist_ok=True)
    marker = study / "native_features/train_complete.json"
    native_spec = json.loads(marker.read_text())
    data, ids = bundle["data"], bundle["data"]["sample_ids"]
    if (
        native_spec["base_digest"] != cfg.base_digest
        or native_spec["views"] != 1
        or native_spec["image_size"] != bundle["config"]["image_size"]
        or not np.array_equal(native_spec["ids"], ids)
    ):
        raise ValueError("Training feature identities differ from base")
    lookup = {int(s): i for i, s in enumerate(ids)}
    fit = np.array([lookup[int(s)] for s in data["meta_train_ids"]])
    branch = study / "projected/base_resnet50"
    if not (branch / "projection.npz").exists():
        raise ValueError("Full source PCA required")
    native = np.load(study / "native_features/train_features.npy", mmap_mode="r")[:, 0]
    projection = fit_projection(
        native, fit, branch / "projection.npz", cfg.feature_dim, cfg.pca_seed
    )
    features = np.load(branch / "train.npy", mmap_mode="r")
    if (
        features.shape != (len(ids), cfg.feature_dim)
        or not np.isfinite(features).all()
        or not np.allclose(
            features[:4],
            (native[:4] - projection["mean"]) @ projection["components"].T,
            atol=1e-4,
            rtol=1e-4,
        )
    ):
        raise ValueError("Projected training feature identity differs")
    started, device = time.monotonic(), choose_device(cfg.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    deadline = started + cfg.max_hours * 3600
    metrics = dict(
        status="running",
        base_retrained=False,
        original_heads_reused=0,
        new_head_count=len(cfg.seeds),
        test_evaluated=False,
        completed_stage="source_verified",
    )
    metadata = dict(
        base_seed=bundle["config"]["seed"],
        device=str(device),
        checkpoint_path=str(source / "checkpoints/best.pt"),
        best_epoch=bundle["checkpoint"]["epoch"],
        best_val_loss=bundle["checkpoint"]["val_loss"],
        source_study=str(study),
        augmentation_distribution="base training transform",
        augmentation_rng="new deterministic draws; original training draws were not recorded",
    )

    def finish(status):
        metrics.update(status=status, elapsed_seconds=time.monotonic() - started)
        export(
            out,
            cfg,
            metrics,
            metadata,
            notes=(
                "Fixed selected checkpoint; five new deterministic draws from the actual training "
                "RandomCrop(32,padding=4,constant) then horizontal-flip distribution. These are not "
                "the original training RNG draws. Only observed labels construct targets. "
                "Existing train-only PCA, image features and nine original heads are reused "
                "(three heads per original target when three seeds are present). Only the augmented "
                "static heads are new. No base retraining. All old validation policies must reproduce; "
                "new policies freeze before test labels are loaded. Paired test-example intervals "
                "are exploratory, conditional on one fitted base and uncorrected for multiplicity. "
                "The control addresses augmentation distribution, not changing online weights, "
                "checkpoint timing or the CE/GCE loss transition. No clean training-label access."
            ),
        )
        return metrics

    def checkpoint(stage):
        metrics["completed_stage"] = stage
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Control time budget exhausted; completed stages can resume"
            )

    old_config = out / "config.yaml"
    if old_config.exists():
        prior = OmegaConf.to_container(OmegaConf.load(old_config), resolve=True)
        ignored = {
            "device",
            "max_hours",
            "min_free_gb",
            "image_batch_size",
            "static_target_digest",
        }
        current = OmegaConf.to_container(cfg, resolve=True)
        if {k: v for k, v in prior.items() if k not in ignored} != {
            k: v for k, v in current.items() if k not in ignored
        }:
            raise ValueError("Existing control scientific configuration differs")
    OmegaConf.save(cfg, old_config, resolve=True)
    try:
        cls = {"cifar10n": CIFAR10, "cifar100n": CIFAR100}[cfg.dataset.name]
        raw = cls(cfg.data_dir, train=True, download=False)
        values = augmented_margin(
            bundle,
            raw.data,
            out / "augmented_target",
            cfg.augmentation_views,
            cfg.augmentation_seed,
            cfg.image_batch_size,
            device,
            deadline,
        )
        cfg.static_target_digest = checkpoint_digest(
            out / "augmented_target/mean_margin.npy"
        )
        data["static_augmented_margin"] = np.asarray(values)
        OmegaConf.save(cfg, old_config, resolve=True)
        with np.load(study / "artifacts/training_targets.npz") as targets:
            if not np.array_equal(targets["sample_ids"], ids) or not np.array_equal(
                targets["noisy_labels"], data["noisy_labels"]
            ):
                raise ValueError("Original static target identities differ")
            best_margin = targets["best_margin"]
        np.savez_compressed(
            out / "artifacts/training_targets.npz", **data, best_margin=best_margin
        )
        checkpoint("augmented_targets_complete")
        for mode in cfg.target_modes:
            for seed in cfg.seeds:
                if mode != "best_augmented_mean":
                    load_feature_head(cfg, study, "base_resnet50", mode, seed, device)
                    old = study / "heads" / f"base_resnet50_{mode}_{seed}"
                    new = out / "heads" / old.name
                    if not new.exists():
                        shutil.copytree(old, new)
                    metrics["original_heads_reused"] += 1
                model = train_feature_head(
                    cfg,
                    bundle,
                    out,
                    features,
                    best_margin,
                    "base_resnet50",
                    mode,
                    seed,
                    device,
                )
                del model
                checkpoint(f"head_{mode}_{seed}")
        reconstruction(
            cfg, bundle, out, {"base_resnet50": features}, best_margin, device
        )
        p, y, val_ids = shared_split(study, post, cfg, "val")
        if not np.array_equal(val_ids, data["base_val_ids"]) or not np.array_equal(
            y, noisy[val_ids]
        ):
            raise ValueError("Validation cache labels differ from observed annotations")
        val_features = np.load(branch / "val.npy", mmap_mode="r")
        if val_features.shape != (*p.shape[:2], cfg.feature_dim):
            raise ValueError("Validation projected feature shape differs")
        frozen = freeze_policies(
            cfg, out, p, y, {"base_resnet50": val_features}, device
        )
        for name, policy in frozen["selected"].items():
            if (
                "best_augmented_mean" not in name
                and policy != original["selected"][name]
            ):
                raise ValueError(
                    "Reused head no longer reproduces its original validation policy"
                )
        checkpoint("all_policies_frozen")
        # Clean test labels are first accessed only after the complete new policy marker.
        p, y, test_ids = shared_split(study, post, cfg, "test")
        test_features = np.load(branch / "test.npy", mmap_mode="r")
        if test_features.shape != (*p.shape[:2], cfg.feature_dim):
            raise ValueError("Test projected feature shape differs")
        metrics["selected_results"], metrics["aggregate"] = evaluate_policies(
            cfg, out, p, y, {"base_resnet50": test_features}, device, frozen
        )
        metrics["test_evaluated"] = True
        contrasts = target_contrasts(
            cfg,
            out,
            p,
            y,
            test_ids,
            frozen,
            cfg.auroc_bootstrap_draws,
            cfg.bootstrap_seed,
        )
        metrics["paired_comparisons"] = len(contrasts)
        metrics["completed_stage"] = "control_and_paired_comparisons_complete"
        return finish("complete")
    except TimeoutError:
        return finish("paused_time_budget")
    except Exception as exc:
        (out / "artifacts/failure.json").write_text(
            json.dumps(dict(error_type=type(exc).__name__, error=str(exc)))
        )
        finish("failed")
        raise


@hydra.main(
    config_path=str(config_directory()),
    config_name="augmentation_control",
    version_base=None,
)
def main(cfg):
    run_control(cfg, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
