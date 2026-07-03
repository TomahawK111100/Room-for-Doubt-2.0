import json
import os
import shutil

import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

from src.dataset import CIFARDataModule, get_train_val_indices
from src.logging_utils import (
    TRAJECTORY_COLUMNS,
    generate_report,
    save_metrics,
    save_run_metadata,
    save_trajectory_batch,
    validate_trajectory_schema,
)
from src.model import ResNetClassifier

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(REPO / "configs")


@pytest.fixture(scope="module")
def dm():
    d = CIFARDataModule(data_dir="./data", batch_size=32, num_workers=0, val_ratio=0.1, seed=42)
    d.prepare_data()
    d.setup()
    return d


@pytest.fixture
def tmp_out(tmp_path):
    yield str(tmp_path)
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_train_val_split(dm):
    assert len(dm.train_set) == 45000
    assert len(dm.val_set) == 5000
    assert len(dm.test_set) == 10000
    assert set(dm.train_set.sample_ids).isdisjoint(set(dm.val_set.sample_ids))


def test_stable_sample_ids():
    a, b = get_train_val_indices(50000, 0.1, 99)
    c, d = get_train_val_indices(50000, 0.1, 99)
    assert (a == c).all() and (b == d).all()


def test_dataloader_format(dm):
    imgs, ny, cl, sid = next(iter(dm.train_dataloader()))
    assert imgs.shape == (32, 3, 32, 32) and ny.shape == cl.shape == sid.shape == (32,)


def test_resnet(dm):
    m = ResNetClassifier(num_classes=10, pretrained=False)
    batch = next(iter(dm.train_dataloader()))
    out = m.training_step(batch, 0)
    assert out["loss"].ndim == 0 and "trajectory" in out


def test_trajectory_schema(tmp_out):
    batch = {
        "per_sample_loss": torch.tensor([0.1, 0.2]),
        "per_sample_probs": torch.tensor([[0.9, 0.1], [0.4, 0.6]]),
        "noisy_labels": torch.tensor([0, 1]),
        "clean_labels": torch.tensor([0, 0]),
        "sample_ids": torch.tensor([1, 2]),
        "batch_idx": 0,
    }
    p = save_trajectory_batch(tmp_out, 0, "train", batch)
    df = pd.read_csv(p)
    validate_trajectory_schema(df)
    assert set(df["split"]) == {"train"}


def test_checkpoint_monitor():
    cb = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    assert cb.monitor == "val_loss" and cb.mode == "min"


def test_run_artifacts(tmp_out):
    cfg = {"dataset": {"name": "cifar10n", "noisy_split": "worse", "val_ratio": 0.1}}
    save_run_metadata(tmp_out, "r1", cfg, seed=42, checkpoint_path="ckpt/best.ckpt")
    save_metrics(tmp_out, {"val_loss": 0.5})
    generate_report(tmp_out, "r1", cfg, {"val_loss": 0.5}, {"checkpoint_path": "ckpt/best.ckpt"})
    assert all((Path(tmp_out) / f).exists() for f in ["run_metadata.json", "metrics.json", "report.md"])


def test_hydra_config():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config")
    assert cfg.model.num_classes == 10
