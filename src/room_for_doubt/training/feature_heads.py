"""Matched feature-only regressors for a fixed encoder-by-target comparison."""

import csv
import json
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from room_for_doubt.models.feature_margin import FeatureMarginHead
from room_for_doubt.training.meta_retraining import fit_head
from room_for_doubt.training.diagnostic_controls import control_target
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
)
from room_for_doubt.utils.reproducibility import set_seed


def target_values(data, best_margin, mode, early_epochs):
    windows = {
        "early_mean_2": (0, 2),
        "early_mean_10": (0, 10),
        "early_mean_20": (0, 20),
        "late_mean_5": (-5, None),
    }
    if mode in windows:
        start, stop = windows[mode]
        scalar = data["targets"][:, start:stop].mean(1, keepdims=True)
        return np.repeat(scalar, 50, axis=1).astype("float32")
    if mode == "history_mean":
        return np.repeat(data["targets"].mean(1, keepdims=True), 50, axis=1).astype(
            "float32"
        )
    return control_target(data, best_margin, mode, early_epochs)


def verify_config(saved, cfg, encoder, mode, seed):
    for key, value in dict(
        encoder=encoder,
        target_mode=mode,
        seed=int(seed),
        base_digest=cfg.base_digest,
        feature_dim=cfg.feature_dim,
        pca_seed=cfg.pca_seed,
        early_epochs=cfg.early_epochs,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        split_seed=cfg.split_seed,
    ).items():
        if saved.get(key) != value:
            raise ValueError("Feature-head scientific config differs: " + key)


def load_feature_head(cfg, out, encoder, mode, seed, device):
    run = Path(out) / "heads" / f"{encoder}_{mode}_{seed}"
    metrics = json.loads((run / "metrics.json").read_text())
    saved = torch.load(
        run / "checkpoints/best.pt", map_location="cpu", weights_only=False
    )
    verify_config(saved["config"], cfg, encoder, mode, seed)
    with (run / "history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    if metrics["epochs_completed"] != cfg.epochs or [
        int(r["epoch"]) for r in history
    ] != list(range(cfg.epochs)):
        raise ValueError("Incomplete feature head")
    best = min(history, key=lambda r: float(r["val_loss"]))
    if saved["epoch"] != int(best["epoch"]) or not np.isclose(
        saved["val_loss"], float(best["val_loss"]), atol=1e-10, rtol=0
    ):
        raise ValueError("Feature head is not minimum meta-validation MSE")
    model = FeatureMarginHead(cfg.dataset.num_classes, cfg.feature_dim)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval()


def train_feature_head(
    cfg, bundle, out, features, best_margin, encoder, mode, seed, device
):
    run = Path(out) / "heads" / f"{encoder}_{mode}_{seed}"
    if (run / "metrics.json").exists():
        return load_feature_head(cfg, out, encoder, mode, seed, device)
    run.mkdir(parents=True, exist_ok=True)
    local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    local.seed, local.encoder, local.target_mode = int(seed), encoder, mode
    local.target_path = str(Path(out) / "artifacts/training_targets.npz")
    local.cache_path = str(Path(out) / "projected" / encoder)
    last = run / "checkpoints/last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        verify_config(saved["config"], cfg, encoder, mode, seed)
        local.resume_path = str(last)
    data = dict(bundle["data"])
    data["targets"] = target_values(data, best_margin, mode, cfg.early_epochs)
    set_seed(seed)
    model = FeatureMarginHead(cfg.dataset.num_classes, cfg.feature_dim)
    OmegaConf.save(local, run / "config.yaml", resolve=True)
    metrics = fit_head(model, data, features, local, run, device)
    meta = dict(
        seed=int(seed),
        checkpoint_path=str(run / "checkpoints/best.pt"),
        best_epoch=metrics["best_epoch"],
        best_val_loss=metrics["best_val_loss"],
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    )
    config = OmegaConf.to_container(local, resolve=True)
    save_run_metadata(run, run.name, config, **meta)
    generate_report(
        run,
        run.name,
        config,
        metrics,
        meta,
        "Feature-only scalar head; same PCA output dimension, architecture, initialization seed and training budget for every encoder/target. No clean-label fitting.",
    )
    np.savez_compressed(
        run / "split_ids.npz", **{k: v for k, v in data.items() if k.endswith("_ids")}
    )
    save_metrics(run, metrics)
    return load_feature_head(cfg, out, encoder, mode, seed, device)
