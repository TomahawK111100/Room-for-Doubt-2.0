"""Closeout jobs must never retrain the old base or repeat completed base seeds."""

import importlib.util
import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c10_closeout", ROOT / "scripts/kaggle_cifar10_closeout.py")
launcher = importlib.util.module_from_spec(spec)
with pytest.MonkeyPatch.context() as patch:
    patch.syspath_prepend(str(ROOT / "scripts"))
    spec.loader.exec_module(launcher)


def test_new_base_is_only44_with_existing_split_and_training_recipe(tmp_path):
    settings = OmegaConf.load(ROOT / "configs/cifar10_closeout.yaml")
    cfg = launcher.base_config(settings, tmp_path / "base", tmp_path / "data")
    assert cfg.seed == 44 and cfg.base_split_seed == 42
    assert cfg.dataset.name == "cifar10n" and cfg.dataset.noisy_split == "worse"
    assert cfg.dataset.num_classes == 10 and cfg.dataset.val_ratio == .1
    assert cfg.epochs == 50 and cfg.loss.ce_warmup_epochs == 5 and cfg.loss.q == .7
    assert cfg.batch_size == 128 and cfg.image_size == 224 and cfg.resume == "auto"
    assert cfg.model.weights == "IMAGENET1K_V2" and cfg.optimizer.lr == .01


def test_rebuilt_heads_keep_matched_original_configuration(tmp_path):
    cfg = launcher.study_config(tmp_path / "base", tmp_path / "study", tmp_path / "data", 3)
    assert cfg.post_run is None and list(cfg.seeds) == [42, 43, 44]
    assert cfg.epochs == 25 and list(cfg.encoders) == ["resnet18", "base_resnet50"]
    assert list(cfg.target_modes) == ["early_mean", "history_mean", "best_margin"]
    assert cfg.feature_dim == 512 and cfg.early_epochs == 5


def test_old_base_discovery_rejects_partial_or_wrong_weights(tmp_path):
    base = tmp_path / "base"
    (base / "checkpoints").mkdir(parents=True)
    (base / "artifacts").mkdir()
    OmegaConf.save(OmegaConf.create(dict(seed=42, epochs=50, model=dict(name="resnet50"),
                                        dataset=dict(name="cifar10n", noisy_split="worse"))), base / "config.yaml")
    (base / "metrics.json").write_text(json.dumps(dict(status="complete", completed_epochs=49)))
    for name in ["checkpoints/best.pt", "history.csv", "artifacts/meta_targets.npz", "artifacts/split_ids.npz"]:
        (base / name).write_bytes(b"test")
    digest = launcher.followups.digest(base / "checkpoints/best.pt")
    assert launcher.matching_bases(tmp_path, digest) == []
    (base / "metrics.json").write_text(json.dumps(dict(status="complete", completed_epochs=50)))
    assert launcher.matching_bases(tmp_path, "wrong") == []
    assert launcher.matching_bases(tmp_path, digest) == [str(base)]


def test_missing_inputs_stop_without_training(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.followups, "inspect_inputs", lambda *args: dict(studies=[]))
    report = launcher.inspect_control42(tmp_path, "digest")
    assert not report["can_start"] and not report["rebuilding_heads"]


def test_input_data_link_reuses_files_without_copy(tmp_path):
    inputs, data = tmp_path / "inputs", tmp_path / "working"
    (inputs / "cifar-10-batches-py").mkdir(parents=True)
    (inputs / "CIFAR-10_human.pt").write_bytes(b"annotations")
    launcher.link_data(inputs, data)
    assert (data / "cifar-10-batches-py").is_symlink()
    assert (data / "CIFAR-10_human.pt").read_bytes() == b"annotations"


def test_reconstruction_differences_are_reported_not_selected(tmp_path, monkeypatch):
    import csv
    original = tmp_path / "repo/paper/evidence/c10_s42"
    original.mkdir(parents=True)
    study = tmp_path / "study/tables"
    study.mkdir(parents=True)
    header = "method,acc,views,temperature\n"
    (original / "test_validation_selected.csv").write_text(header + "base,0.8,1,uniform\n")
    (study / "test_validation_selected.csv").write_text(header + "base,0.79,1,uniform\n")
    monkeypatch.setattr(launcher, "ROOT", tmp_path / "repo")
    target = tmp_path / "comparison.csv"
    launcher.reconstruction_table(study.parent, target)
    with target.open() as stream:
        row = next(csv.DictReader(stream))
    assert float(row["gain_pp"]) == pytest.approx(-1.)
    assert row["original_acc"] == "0.8" and row["rebuilt_acc"] == "0.79"
