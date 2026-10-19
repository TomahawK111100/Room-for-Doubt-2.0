"""Benchmark frozen inference on observed-validation images without accuracy selection."""

import json
import platform
from pathlib import Path
import hydra
import numpy as np
import torch
from PIL import Image
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from torchvision.datasets import CIFAR10, CIFAR100
from room_for_doubt.data.cifar_noisy import load_observed_labels
from room_for_doubt.data.feature_artifacts import completed_study
from room_for_doubt.data.kaggle_artifacts import load_completed_run, restore_base
from room_for_doubt.data.observed_images import imagenet_transform
from room_for_doubt.evaluation.campaign_cache import make_views
from room_for_doubt.evaluation.runtime_cost import measure
from room_for_doubt.inference.classifier_features import fit_projection
from room_for_doubt.inference.runtime_tta import FrozenViewPredictor
from room_for_doubt.training.feature_heads import load_feature_head
from room_for_doubt.training.meta_retraining import choose_device
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
    write_csv,
)
from room_for_doubt.utils.paths import config_directory


def run_benchmark(cfg, out):
    if not cfg.source_run or not cfg.study_run:
        raise ValueError("Full base and feature-study Outputs required")
    source, study, out = map(
        lambda p: Path(p).resolve(), (cfg.source_run, cfg.study_run, out)
    )
    if any(out == p or out in p.parents or p in out.parents for p in (source, study)):
        raise ValueError("Keep benchmark output separate from inputs")
    saved, frozen = completed_study(study)
    if (
        cfg.head_seed not in saved.seeds
        or cfg.cpu_threads < 1
        or not cfg.batch_images
        or not cfg.view_counts
        or list(cfg.view_counts) != sorted(set(cfg.view_counts))
        or list(cfg.batch_images) != sorted(set(cfg.batch_images))
        or min(cfg.batch_images) < 1
        or min(cfg.view_counts) < 1
        or 1 not in cfg.view_counts
        or max(cfg.batch_images) * max(cfg.view_counts) > 64
        or any(v not in saved.view_counts for v in cfg.view_counts)
    ):
        raise ValueError(
            "Use saved view budgets, an existing head seed and at most 64 expanded views"
        )
    original = OmegaConf.load(source / "config.yaml")
    noisy = load_observed_labels(
        cfg.data_dir, original.dataset.name, original.dataset.noisy_split
    )
    bundle = load_completed_run(source, noisy)
    if saved.base_digest != bundle["checkpoint_digest"]:
        raise ValueError("Benchmark base and feature study differ")
    ids = bundle["data"]["base_val_ids"][: max(cfg.batch_images)]
    if len(ids) != max(cfg.batch_images):
        raise ValueError("Insufficient validation images for requested batch")
    feature_source = (
        Path(saved.study_run) if saved.get("scope") == "augmentation_control" else study
    )
    branch = feature_source / "projected/base_resnet50"
    native_spec = json.loads(
        (feature_source / "native_features/train_complete.json").read_text()
    )
    if native_spec["base_digest"] != saved.base_digest or not np.array_equal(
        native_spec["ids"], bundle["data"]["sample_ids"]
    ):
        raise ValueError("PCA feature identities differ")
    lookup = {int(s): i for i, s in enumerate(bundle["data"]["sample_ids"])}
    fit = np.array([lookup[int(s)] for s in bundle["data"]["meta_train_ids"]])
    if not (branch / "projection.npz").exists():
        raise ValueError("Require the original train-only PCA")
    native = np.load(
        feature_source / "native_features/train_features.npy", mmap_mode="r"
    )[:, 0]
    projection = fit_projection(
        native, fit, branch / "projection.npz", saved.feature_dim, saved.pca_seed
    )
    device = choose_device(cfg.device)
    torch.set_num_threads(cfg.cpu_threads)
    base = restore_base(bundle, device)
    head = load_feature_head(
        saved, study, "base_resnet50", "early_mean", cfg.head_seed, "cpu"
    )
    cls = {"cifar10n": CIFAR10, "cifar100n": CIFAR100}[original.dataset.name]
    raw = cls(cfg.data_dir, train=True, download=False)
    transform = imagenet_transform(original.image_size)
    effective = OmegaConf.merge(
        OmegaConf.to_container(cfg, resolve=True),
        dict(
            dataset=bundle["config"]["dataset"],
            base_digest=saved.base_digest,
            base_seed=bundle["config"]["seed"],
            sample_ids=ids.tolist(),
            precision="fp32",
        ),
    )
    for name in ("tables", "artifacts"):
        (out / name).mkdir(parents=True, exist_ok=True)
    if (out / "config.yaml").exists():
        raise ValueError("Use a new run_id for independent timing repetitions")
    OmegaConf.save(effective, out / "config.yaml", resolve=True)
    name = f"base_resnet50__early_mean__{cfg.head_seed}"
    rows, samples = [], {}
    for batch in cfg.batch_images:
        for count in cfg.view_counts:
            policy = next(
                p
                for p in frozen["policies"]
                if p["method"] == name and p["views"] == count
            )
            for method in (
                ["base", "uniform", "early"] if count == 1 else ["uniform", "early"]
            ):
                active = (
                    method == "early"
                    and count > 1
                    and policy["temperature"] != "uniform"
                )
                predictor = (
                    FrozenViewPredictor(
                        base, head if active else None, projection if active else None
                    )
                    .to(device)
                    .eval()
                )

                def prepare():
                    tensors = [
                        transform(Image.fromarray(v))
                        for sid in ids[:batch]
                        for v in make_views(
                            raw.data[int(sid)], count, saved.inference_seed + int(sid)
                        )
                    ]
                    return (
                        torch.stack(tensors)
                        .reshape(
                            batch, count, 3, original.image_size, original.image_size
                        )
                        .to(device)
                    )

                for scope in ("model_only", "with_preprocessing"):
                    staged = prepare() if scope == "model_only" else None

                    def predict(
                        predictor=predictor,
                        staged=staged,
                        scope=scope,
                        method=method,
                        temperature=policy["temperature"],
                        prepare=prepare,
                    ):
                        return predictor(
                            staged if scope == "model_only" else prepare(),
                            method,
                            temperature,
                        )

                    result, times = measure(
                        predict, device, cfg.warmup, cfg.repetitions, batch
                    )
                    key = f"{method}_{count}_{batch}_{scope}"
                    samples[key] = times
                    rows.append(
                        dict(
                            method=method,
                            views=count,
                            batch_images=batch,
                            scope=scope,
                            head_used=active,
                            temperature=policy["temperature"]
                            if method == "early"
                            else "uniform",
                            warmup=cfg.warmup,
                            repetitions=cfg.repetitions,
                            **result,
                        )
                    )
                    del predict, staged
                del predictor
                head.to("cpu")
    write_csv(out / "tables/runtime_cost.csv", rows)
    np.savez_compressed(out / "artifacts/timings_ms.npz", **samples)
    metadata = dict(
        device=str(device),
        device_name=torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.machine(),
        torch_version=torch.__version__,
        cpu_threads=torch.get_num_threads(),
        cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        checkpoint_path=str(source / "checkpoints/best.pt"),
        best_epoch=bundle["checkpoint"]["epoch"],
        best_val_loss=bundle["checkpoint"]["val_loss"],
    )
    metrics = dict(
        status="complete",
        configurations=len(rows),
        base_retrained=False,
        head_retrained=False,
        clean_test_accessed=False,
    )
    config = OmegaConf.to_container(effective, resolve=True)
    save_metrics(out, metrics)
    save_run_metadata(out, cfg.run_id, config, **metadata)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        metadata,
        "Final-probability consensus only, not standalone error-ranking score latency. "
        "Fused FP32 reference deployment, not research cache-generation time. Same validation images "
        "and deterministic view prefixes; no labels used to choose timing cases. Model-only excludes "
        "CPU transforms and host-to-device transfer; with_preprocessing includes both. Disk loading and "
        "model initialization are excluded. CUDA timings synchronize before and after each repeat. "
        "Batch p95 is not single-image p95 for batches larger than one. CUDA memory is PyTorch allocated "
        "memory, not total device usage; CPU/MPS peak memory is unavailable. Uniform and single-view "
        "policies skip an unnecessary head. Each method reports its own resident allocations. "
        "Only an otherwise idle device gives comparable timings; report hardware with results.",
    )
    return metrics


@hydra.main(
    config_path=str(config_directory()), config_name="runtime_cost", version_base=None
)
def main(cfg):
    run_benchmark(cfg, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
