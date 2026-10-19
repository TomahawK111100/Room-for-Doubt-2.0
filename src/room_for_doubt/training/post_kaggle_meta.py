"""Train and validate new-base meta heads; never reuse historical heads."""

import csv
import json
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from room_for_doubt.models.conditioned_trajectory import ConditionedPredictor
from room_for_doubt.models.scalar_margin import ScalarMarginPredictor
from room_for_doubt.training.meta_retraining import fit_head
from room_for_doubt.utils.reproducibility import set_seed
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
)


def head_config(cfg, bundle, mode, seed, root):
    local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    local.seed = seed
    local.target_mode = mode
    local.base_digest = bundle["checkpoint_digest"]
    local.dataset = bundle["config"]["dataset"]
    local.target_path = str(Path(root) / "artifacts/meta_targets.npz")
    local.cache_path = str(Path(root) / "features")
    local.run_id = f"{cfg.run_id}_{mode}_{seed}"
    return local


def train_heads(cfg, bundle, root, device):
    root = Path(root)
    with np.load(root / "features/features.npz") as f:
        if not np.array_equal(f["sample_ids"], bundle["data"]["sample_ids"]):
            raise ValueError("Feature rows differ from source IDs")
        features = f["features"]
        if (
            f["dataset_name"].item() != bundle["config"]["dataset"]["name"]
            or f["preprocessing"].item() != "imagenet_pil_bilinear_224"
            or features.shape != (len(bundle["data"]["sample_ids"]), 512)
            or not np.isfinite(features).all()
        ):
            raise ValueError("Invalid shared meta-feature cache")
    backbone = torch.load(
        root / "features/backbone.pt", map_location="cpu", weights_only=True
    )
    for mode in cfg.modes:
        if mode not in {"mean", "full"}:
            raise ValueError("Supported target modes: mean/full")
        data = dict(bundle["data"])
        if mode == "mean":
            data["targets"] = np.repeat(
                data["targets"].mean(1, keepdims=True), 50, axis=1
            )
        for seed in cfg.seeds:
            local = head_config(cfg, bundle, mode, seed, root)
            run = root / "heads" / f"{mode}_{seed}"
            if (run / "metrics.json").exists():
                load_head(cfg, bundle, mode, seed, root, device)
                print("Reuse completed head", mode, seed, flush=True)
                continue
            run.mkdir(parents=True, exist_ok=True)
            set_seed(seed)
            model = (ScalarMarginPredictor if mode == "mean" else ConditionedPredictor)(
                bundle["config"]["dataset"]["num_classes"], pretrained=False
            )
            model.features.load_state_dict(backbone)
            if (run / "checkpoints/last.pt").exists():
                local.resume_path = str(run / "checkpoints/last.pt")
            # fit_head validates its scientific resume fields. resume_path itself
            # is deliberately excluded by that existing validation routine.
            OmegaConf.save(local, run / "config.yaml", resolve=True)
            metrics = fit_head(model, data, features, local, run, device)
            np.savez_compressed(
                run / "split_ids.npz",
                **{k: v for k, v in bundle["data"].items() if k.endswith("_ids")},
            )
            meta = dict(
                seed=seed,
                device=str(device),
                checkpoint_path=str(run / "checkpoints/best.pt"),
                base_digest=bundle["checkpoint_digest"],
                best_epoch=metrics["best_epoch"],
                best_val_loss=metrics["best_val_loss"],
            )
            config = OmegaConf.to_container(local, resolve=True)
            save_run_metadata(run, local.run_id, config, **meta)
            generate_report(
                run,
                local.run_id,
                config,
                metrics,
                meta,
                "New-base observed-class targets only; held-out meta MSE checkpoint selection. No test evaluation or clean training-label supervision. Scalar mean and full heads share frozen inputs and hidden width, with different output dimensions.",
            )
            # A metrics file marks the fully exported head, not a partial epoch.
            save_metrics(run, metrics)


def load_head(cfg, bundle, mode, seed, root, device):
    root = Path(root)
    run = root / "heads" / f"{mode}_{seed}"
    metrics = json.loads((run / "metrics.json").read_text())
    if metrics["epochs_completed"] != cfg.epochs:
        raise ValueError("Incomplete meta head")
    best = torch.load(
        run / "checkpoints/best.pt", map_location="cpu", weights_only=False
    )
    expected = OmegaConf.to_container(
        head_config(cfg, bundle, mode, seed, root), resolve=True
    )
    for key in [
        "base_digest",
        "target_mode",
        "seed",
        "dataset",
        "epochs",
        "batch_size",
        "learning_rate",
        "split_seed",
        "meta_model",
    ]:
        if best["config"].get(key) != expected[key]:
            raise ValueError(f"Meta head config mismatch: {key}")
    with (run / "history.csv").open() as f:
        history = list(csv.DictReader(f))
    if [int(r["epoch"]) for r in history] != list(range(cfg.epochs)):
        raise ValueError("Incomplete meta validation history")
    best_row = min(history, key=lambda r: float(r["val_loss"]))
    if best["epoch"] != int(best_row["epoch"]) or not np.isclose(
        best["val_loss"], float(best_row["val_loss"]), rtol=0, atol=1e-10
    ):
        raise ValueError("Meta checkpoint is not minimum validation MSE")
    with np.load(run / "split_ids.npz") as f:
        for name in bundle["data"]:
            if name.endswith("_ids") and not np.array_equal(
                f[name], bundle["data"][name]
            ):
                raise ValueError("Meta partition mismatch")
    model = (ScalarMarginPredictor if mode == "mean" else ConditionedPredictor)(
        bundle["config"]["dataset"]["num_classes"], pretrained=False
    )
    model.load_state_dict(best["state_dict"], strict=True)
    backbone = torch.load(
        root / "features/backbone.pt", map_location="cpu", weights_only=True
    )
    if any(
        not torch.equal(v, model.features.state_dict()[k]) for k, v in backbone.items()
    ):
        raise ValueError("Meta encoder differs from shared features")
    return model.to(device).eval()
