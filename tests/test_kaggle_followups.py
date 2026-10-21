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
    for split in ("val", "test"):
        (study / "shared/cache" / f"{split}_complete.json").write_text(
            json.dumps(dict(base_digest=sha, seed=20261015, views=10))
        )
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


def test_policy_digest_is_reported_and_blocks_execution(tmp_path, monkeypatch):
    study, _, sha = make_study(tmp_path)
    monkeypatch.setattr(launcher, "EXPECTED", dict(cifar100n_seed44=sha))
    (study / "artifacts/policies_before_test.json").write_text(
        json.dumps(dict(base_digest="different model"))
    )
    row = launcher.inspect_inputs(tmp_path)["studies"][0]
    assert row["missing_files"] == []
    assert row["digest_matches"]
    assert not row["valid"]
    assert "Frozen policy digest differs from study" in row["rejection_reasons"]
    with pytest.raises(ValueError, match="Frozen policy digest differs"):
        launcher.discover(tmp_path)


def test_same_preflight_for_inspection_and_execution(tmp_path, monkeypatch):
    _, _, sha = make_study(tmp_path)
    monkeypatch.setattr(launcher, "EXPECTED", dict(cifar100n_seed44=sha))
    original = Path.rglob

    def single_pattern_only(path, pattern, *args, **kwargs):
        assert "/" not in pattern, "Do not use compound glob to discover markers"
        return original(path, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "rglob", single_pattern_only)
    row = launcher.inspect_inputs(tmp_path)["studies"][0]
    assert row["valid"] and row["checkpoint_digest"] == sha
    assert launcher.discover(tmp_path)[0]["study"] == row["study"]


def make_external_study(root):
    import shutil
    study, base, sha = make_study(root / "study_output")
    model = root / "base_output/source_model"
    model.parent.mkdir()
    shutil.move(str(base), str(model))
    OmegaConf.save(OmegaConf.create(dict(
        model=dict(name="resnet50"), epochs=50, seed=44,
        dataset=dict(name="cifar100n"),
    )), model / "config.yaml")
    post = root / "post_output/original_post"
    post.parent.mkdir()
    shutil.move(str(study / "shared"), str(post))
    cfg = OmegaConf.load(study / "config.yaml")
    cfg.source_run = "/previous_session/nonexistent_model"
    cfg.post_run = "/previous_session/nonexistent_post"
    cfg.inference_seed = 20261015
    cfg.view_counts = [1, 5, 10]
    OmegaConf.save(cfg, study / "config.yaml")
    for split in ("val", "test"):
        (post / "cache" / f"{split}_complete.json").write_text(json.dumps(
            dict(base_digest=sha, seed=20261015, views=10)
        ))
    return study, model, post, sha


def test_external_base_and_post_resolve_by_identity(tmp_path, monkeypatch):
    study, model, post, sha = make_external_study(tmp_path)
    monkeypatch.setattr(launcher, "EXPECTED", dict(cifar100n_seed44=sha))
    row = launcher.discover(tmp_path)[0]
    assert row["study"] == str(study)
    assert row["base"] == str(model)
    assert row["post"] == str(post)


def test_recovered_cache_outside_inputs_resolves(tmp_path, monkeypatch):
    import shutil
    inputs = tmp_path / "input"
    study, model, post, sha = make_external_study(inputs)
    recovered = tmp_path / "working/recovered"
    recovered.parent.mkdir()
    shutil.move(str(post), str(recovered))
    (study / "shared/cache").mkdir(parents=True)
    monkeypatch.setattr(launcher, "EXPECTED", dict(cifar100n_seed44=sha))
    row = launcher.discover(inputs, cache_root=recovered.parent)[0]
    assert row["study"] == str(study) and row["base"] == str(model)
    assert row["post"] == str(recovered)


def test_wrong_external_cache_seed_rejected(tmp_path, monkeypatch):
    _, _, post, sha = make_external_study(tmp_path)
    monkeypatch.setattr(launcher, "EXPECTED", dict(cifar100n_seed44=sha))
    path = post / "cache/val_complete.json"
    data = json.loads(path.read_text())
    data["seed"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Missing complete Outputs"):
        launcher.discover(tmp_path)


def test_ambiguous_external_post_rejected(tmp_path, monkeypatch):
    import shutil
    _, _, post, sha = make_external_study(tmp_path)
    shutil.copytree(post, tmp_path / "duplicate_post")
    monkeypatch.setattr(launcher, "EXPECTED", dict(cifar100n_seed44=sha))
    with pytest.raises(ValueError, match="Ambiguous shared cache"):
        launcher.discover(tmp_path)

def test_internal_cache_provenance_checked_before_computation(tmp_path, monkeypatch):
    study, _, sha = make_study(tmp_path)
    monkeypatch.setattr(launcher, "EXPECTED", dict(cifar100n_seed44=sha))
    path = study / "shared/cache/val_complete.json"
    protocol = json.loads(path.read_text())
    protocol["base_digest"] = "different model"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="val shared cache provenance differs"):
        launcher.discover(tmp_path)

def test_data_links_recover_after_input_mount_changes(tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "cifar-10-batches-py").mkdir()
    (inputs / "CIFAR-10_human.pt").write_bytes(b"observed annotations")
    data = tmp_path / "data"
    data.mkdir()
    (data / "CIFAR-10_human.pt").symlink_to(tmp_path / "removed_input")
    launcher.prepare_data(inputs, data, [dict(dataset="cifar10n")])
    assert (data / "CIFAR-10_human.pt").read_bytes() == b"observed annotations"
    assert (data / "cifar-10-batches-py").is_dir()
