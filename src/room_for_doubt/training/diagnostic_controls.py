"""Matched scalar controls using saved frozen features and reserved splits."""

import json
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from room_for_doubt.models.scalar_margin import ScalarMarginPredictor
from room_for_doubt.training.meta_retraining import fit_head
from room_for_doubt.utils.reproducibility import set_seed
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
)


def control_target(data, best_margin, mode, early_epochs):
    if mode == "best_margin":
        scalar = best_margin
    elif mode == "last_margin":
        scalar = data["targets"][:, -1]
    elif mode == "early_mean":
        scalar = data["targets"][:, :early_epochs].mean(1)
    else:
        raise ValueError("Unknown control target")
    return np.repeat(np.asarray(scalar)[:, None], 50, axis=1).astype("float32")


def load_control(run, cfg, mode, seed, device):
    run = Path(run)
    saved = torch.load(
        run / "checkpoints/best.pt", map_location="cpu", weights_only=False
    )
    metrics = json.loads((run / "metrics.json").read_text())
    if metrics["epochs_completed"] != cfg.epochs:
        raise ValueError("Incomplete diagnostic control")
    for key, expected in dict(
        control_mode=mode,
        seed=int(seed),
        base_digest=cfg.base_digest,
        epochs=cfg.epochs,
        early_epochs=cfg.early_epochs,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        split_seed=cfg.split_seed,
    ).items():
        if saved["config"].get(key) != expected:
            raise ValueError("Diagnostic control config differs: " + key)
    model = ScalarMarginPredictor(cfg.dataset.num_classes, pretrained=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval()


def train_control(cfg, bundle, post, out, features, best_margin, mode, seed, device):
    run = Path(out) / "heads" / f"{mode}_{seed}"
    if (run / "metrics.json").exists():
        load_control(run, cfg, mode, seed, device)
        print("Reuse diagnostic control", mode, seed, flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    local.seed, local.control_mode = int(seed), mode
    local.target_path = str(Path(out) / "artifacts/best_train_predictions.npz")
    local.cache_path = str(Path(post) / "features")
    local.resume_path = (
        str(run / "checkpoints/last.pt")
        if (run / "checkpoints/last.pt").exists()
        else None
    )
    data = dict(bundle["data"])
    data["targets"] = control_target(data, best_margin, mode, cfg.early_epochs)
    set_seed(seed)
    model = ScalarMarginPredictor(cfg.dataset.num_classes, pretrained=False)
    model.features.load_state_dict(
        torch.load(
            Path(post) / "features/backbone.pt", map_location="cpu", weights_only=True
        )
    )
    OmegaConf.save(local, run / "config.yaml", resolve=True)
    metrics = fit_head(model, data, features, local, run, device)
    meta = dict(
        seed=int(seed),
        checkpoint_path=str(run / "checkpoints/best.pt"),
        best_epoch=metrics["best_epoch"],
        best_val_loss=metrics["best_val_loss"],
    )
    config = OmegaConf.to_container(local, resolve=True)
    np.savez_compressed(
        run / "split_ids.npz", **{k: v for k, v in data.items() if k.endswith("_ids")}
    )
    save_run_metadata(run, f"{mode}_{seed}", config, **meta)
    generate_report(
        run,
        f"{mode}_{seed}",
        config,
        metrics,
        meta,
        "Frozen source features, observed-class regression, identical scalar architecture and held-out meta MSE checkpoint selection. No clean-label supervision.",
    )
    save_metrics(run, metrics)
