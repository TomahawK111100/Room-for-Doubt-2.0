"""Validate a completed stronger-base run without using clean training labels."""

import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from room_for_doubt.data.cifar_noisy import get_train_val_indices
from room_for_doubt.data.meta_splits import validate_meta_splits, reserve_meta_splits
from room_for_doubt.training.kaggle_base import resume_contract, base_split_seed
from room_for_doubt.models.robust_classifier import make_pretrained_classifier


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_completed_run(source, observed_labels):
    source = Path(source)
    config = OmegaConf.to_container(
        OmegaConf.load(source / "config.yaml"), resolve=True
    )
    status = json.loads((source / "metrics.json").read_text())
    if (
        status.get("status") != "complete"
        or status.get("completed_epochs") != 50
        or config["epochs"] != 50
    ):
        raise ValueError("Require a completed 50-epoch base run, not a partial run")
    if config["model"]["name"] != "resnet50" or config["image_size"] < 1:
        raise ValueError("Unsupported base architecture or preprocessing")
    classes = {"cifar10n": 10, "cifar100n": 100}
    if classes.get(config["dataset"]["name"]) != config["dataset"]["num_classes"]:
        raise ValueError("Source dataset/class-count mismatch")
    checkpoint_path = source / "checkpoints/best.pt"
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if resume_contract(best["config"]) != resume_contract(config):
        raise ValueError("Best checkpoint scientific config differs from source run")
    with (source / "history.csv").open() as f:
        history = list(csv.DictReader(f))
    if [int(r["epoch"]) for r in history] != list(range(50)):
        raise ValueError("Require complete ordered base history for epochs 0..49")
    losses = np.array([float(r["val_loss"]) for r in history])
    epoch = int(losses.argmin())
    if (
        not np.isfinite(losses).all()
        or best["epoch"] != epoch
        or not np.isclose(best["val_loss"], losses[epoch], rtol=0, atol=1e-10)
    ):
        raise ValueError("Best checkpoint does not match minimum noisy validation loss")
    if status["best_epoch"] != epoch or not np.isclose(
        status["best_val_loss"], losses[epoch], rtol=0, atol=1e-10
    ):
        raise ValueError("Source metrics disagree with selected checkpoint")
    with np.load(source / "artifacts/meta_targets.npz", allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    noisy = np.asarray(observed_labels)
    if (
        noisy.ndim != 1
        or np.any(noisy < 0)
        or np.any(noisy >= config["dataset"]["num_classes"])
    ):
        raise ValueError("Invalid observed labels")
    train, val = get_train_val_indices(
        len(noisy), config["dataset"]["val_ratio"], base_split_seed(config)
    )
    for name, expected in [
        ("sample_ids", train),
        ("base_train_ids", train),
        ("base_val_ids", val),
        ("epochs", np.arange(50)),
    ]:
        if not np.array_equal(data[name], expected):
            raise ValueError(f"Source identity mismatch: {name}")
    if (
        data["dataset_name"].item() != config["dataset"]["name"]
        or data["noisy_split"].item() != config["dataset"]["noisy_split"]
    ):
        raise ValueError("Target dataset/annotation mismatch")
    if (
        data["targets"].shape != (len(train), 50)
        or not np.isfinite(data["targets"]).all()
        or np.any(np.abs(data["targets"]) > 1.00001)
    ):
        raise ValueError("Invalid or incomplete probability-margin targets")
    if len(data["excluded_ids"]) or not np.array_equal(
        data["noisy_labels"], noisy[train]
    ):
        raise ValueError("Unexpected exclusions or observed-label mismatch")
    with np.load(source / "artifacts/split_ids.npz") as f:
        if not np.array_equal(f["train_ids"], train) or not np.array_equal(
            f["val_ids"], val
        ):
            raise ValueError("Source split export differs from trajectories")
    validate_meta_splits(data, train)
    if int(data["split_seed"]) != config["split_seed"]:
        raise ValueError("Meta split seed mismatch")
    for name, expected in reserve_meta_splits(train, config["split_seed"]).items():
        if not np.array_equal(data[name], expected):
            raise ValueError("Unexpected reserved meta partition")
    return dict(
        source=source.resolve(),
        config=config,
        data=data,
        checkpoint=best,
        checkpoint_digest=checkpoint_digest(checkpoint_path),
    )


def restore_base(bundle, device):
    model = make_pretrained_classifier(
        bundle["config"]["dataset"]["num_classes"], weights=None
    )
    model.load_state_dict(bundle["checkpoint"]["state_dict"], strict=True)
    return model.to(device).eval()
