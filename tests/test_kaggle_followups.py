"""Full-output discovery must never confuse compact snapshots or base identities."""

import importlib.util
import json
from pathlib import Path
import pytest
from omegaconf import OmegaConf

spec = importlib.util.spec_from_file_location(
    "followups", Path(__file__).resolve().parents[1] / "scripts/kaggle_followups.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def make_study(root):
    study = root / "jobs/cifar100n_seed44/feature_study"
    files = [
        "native_features/train_complete.json",
        "native_features/train_features.npy",
        "artifacts/training_targets.npz",
        "projected/base_resnet50/projection.npz",
        "projected/base_resnet50/train.npy",
        "projected/base_resnet50/val.npy",
        "projected/base_resnet50/test.npy",
    ]
    for split in ("val", "test"):
        files += [f"native_features/{split}_complete.json",
                  f"shared/cache/{split}_complete.json"]
        files += [f"shared/cache/{split}_{suffix}.npy" for suffix in ("probs", "ids", "labels")]
    files += ["heads/base_resnet50_early_mean_42/checkpoints/best.pt",
              "heads/base_resnet50_early_mean_42/history.csv",
              "heads/base_resnet50_early_mean_42/metrics.json",
              "artifacts/test_base_resnet50__early_mean__42.npy"]
    for name in files:
        path = study / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    base = study.parent / "base"
    for name in ("checkpoints/best.pt", "config.yaml", "metrics.json", "history.csv",
                 "artifacts/meta_targets.npz", "artifacts/split_ids.npz"):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic checkpoint")
    sha = launcher.digest(base / "checkpoints/best.pt")
    OmegaConf.save(OmegaConf.create(dict(
        dataset=dict(name="cifar100n"), base_seed=44, base_digest=sha,
        encoders=["base_resnet50"], target_modes=["early_mean"], seeds=[42],
    )), study / "config.yaml")
    (study / "metrics.json").write_text(json.dumps(dict(status="complete")))
    (study / "artifacts/policies_before_test.json").write_text(json.dumps(dict(base_digest=sha)))
    return study, base, sha


def test_exact_base_and_missing_full_output(tmp_path, monkeypatch):
    study, base, sha = make_study(tmp_path)
    monkeypatch.setattr(launcher, "EXPECTED", dict(cifar100n_seed44=sha))
    rows = launcher.discover(tmp_path)
    assert rows[0]["base"] == str(base)
    assert rows[0]["study"] == str(study)
    (study / "projected/base_resnet50/train.npy").unlink()
    with pytest.raises(ValueError, match="Missing complete Outputs"):
        launcher.discover(tmp_path)


def test_wrong_checkpoint_rejected(tmp_path, monkeypatch):
    _, base, sha = make_study(tmp_path)
    monkeypatch.setattr(launcher, "EXPECTED", dict(cifar100n_seed44=sha))
    (base / "checkpoints/best.pt").write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="Checkpoint identity differs"):
        launcher.discover(tmp_path)


def test_duplicate_copies_rejected(tmp_path, monkeypatch):
    _, _, sha = make_study(tmp_path / "a")
    make_study(tmp_path / "b")
    monkeypatch.setattr(launcher, "EXPECTED", dict(cifar100n_seed44=sha))
    with pytest.raises(ValueError, match="Duplicate full Outputs"):
        launcher.discover(tmp_path)


def test_inspection_names_missing_artifact(tmp_path, monkeypatch):
    study, _, sha = make_study(tmp_path)
    monkeypatch.setattr(launcher, "EXPECTED", dict(cifar100n_seed44=sha))
    missing = "projected/base_resnet50/train.npy"
    (study / missing).unlink()
    report = launcher.inspect_inputs(tmp_path)
    assert report["studies"][0]["digest_matches"]
    assert report["studies"][0]["missing_files"] == [missing]
    with pytest.raises(ValueError, match="projected/base_resnet50/train.npy"):
        launcher.discover(tmp_path)


def test_inspection_distinguishes_wrong_base(tmp_path, monkeypatch):
    study, _, _ = make_study(tmp_path)
    monkeypatch.setattr(launcher, "EXPECTED", dict(cifar100n_seed44="other digest"))
    report = launcher.inspect_inputs(tmp_path)
    assert report["studies"][0]["study"] == str(study)
    assert not report["studies"][0]["digest_matches"]
    assert report["studies"][0]["missing_files"] == []
