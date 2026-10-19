import csv
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from room_for_doubt.data.cifar_noisy import get_train_val_indices
from room_for_doubt.data.meta_splits import reserve_meta_splits
from room_for_doubt.data.kaggle_artifacts import load_completed_run

ROOT = Path(__file__).parents[1]


class TinyBase(torch.nn.Module):
    def __init__(self, classes=100, weights=None):
        super().__init__()
        self.fc = torch.nn.Linear(3, classes)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.zero_()
            self.fc.bias[0] = 5

    def forward(self, x):
        return self.fc(x.mean((2, 3)))


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "checkpoints").mkdir(parents=True)
    (root / "artifacts").mkdir()
    cfg = OmegaConf.load(ROOT / "configs/kaggle_base.yaml")
    cfg.image_size = 8
    config = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(cfg, root / "config.yaml")
    best = dict(
        state_dict=TinyBase().state_dict(), epoch=7, val_loss=0.2, config=config
    )
    torch.save(best, root / "checkpoints/best.pt")
    (root / "metrics.json").write_text(
        json.dumps(
            dict(
                status="complete", completed_epochs=50, best_epoch=7, best_val_loss=0.2
            )
        )
    )
    with (root / "history.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "val_loss"])
        w.writeheader()
        for i in range(50):
            w.writerow(dict(epoch=i, val_loss=0.2 if i == 7 else 0.5 + i / 100))
    noisy = np.arange(40) % 100
    train, val = get_train_val_indices(40, 0.1, 42)
    data = dict(
        sample_ids=train,
        base_train_ids=train,
        base_val_ids=val,
        noisy_labels=noisy[train],
        targets=np.random.default_rng(1)
        .uniform(-1, 1, (len(train), 50))
        .astype("float32"),
        epochs=np.arange(50),
        excluded_ids=np.array([], dtype=np.int64),
        dataset_name=np.array("cifar100n"),
        noisy_split=np.array("noisy100"),
        split_seed=np.array(20261015),
        **reserve_meta_splits(train, 20261015),
    )
    np.savez_compressed(root / "artifacts/meta_targets.npz", **data)
    np.savez_compressed(root / "artifacts/split_ids.npz", train_ids=train, val_ids=val)
    return root, noisy


def test_source_rejects_partial_and_wrong_checkpoint(source):
    root, noisy = source
    assert load_completed_run(root, noisy)["checkpoint"]["epoch"] == 7
    p = root / "metrics.json"
    m = json.loads(p.read_text())
    m["status"] = "paused_time_budget"
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="completed 50-epoch"):
        load_completed_run(root, noisy)
    m["status"] = "complete"
    p.write_text(json.dumps(m))
    p = root / "checkpoints/best.pt"
    best = torch.load(p, weights_only=False)
    best["epoch"] = 8
    torch.save(best, p)
    with pytest.raises(ValueError, match="minimum noisy validation"):
        load_completed_run(root, noisy)


def test_source_accepts_legacy_config_and_fixed_split_with_new_model_seed(source):
    root, noisy = source
    cfg = OmegaConf.load(root / "config.yaml")
    del cfg.base_split_seed
    checkpoint = torch.load(root / "checkpoints/best.pt", weights_only=False)
    checkpoint["config"] = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(cfg, root / "config.yaml")
    torch.save(checkpoint, root / "checkpoints/best.pt")
    original = load_completed_run(root, noisy)["data"]["base_train_ids"].copy()
    cfg.seed, cfg.base_split_seed = 43, 42
    checkpoint["config"] = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(cfg, root / "config.yaml")
    torch.save(checkpoint, root / "checkpoints/best.pt")
    assert np.array_equal(
        load_completed_run(root, noisy)["data"]["base_train_ids"], original
    )


def test_source_rejects_missing_epochs_and_wrong_annotations(source):
    root, noisy = source
    bad = noisy.copy()
    bad[0] = 99
    with pytest.raises(ValueError, match="observed-label"):
        load_completed_run(root, bad)
    p = root / "artifacts/meta_targets.npz"
    with np.load(p) as f:
        data = {k: f[k] for k in f.files}
    data["epochs"] = np.arange(1, 51)
    np.savez_compressed(p, **data)
    with pytest.raises(ValueError, match="epochs"):
        load_completed_run(root, noisy)


def test_end_to_end_pipeline_has_no_test_before_frozen_policies(
    source, tmp_path, monkeypatch
):
    import room_for_doubt.analysis.post_kaggle as runner
    import room_for_doubt.data.kaggle_artifacts as artifacts
    import room_for_doubt.inference.post_kaggle_cache as caching
    import room_for_doubt.training.post_kaggle_meta as meta_training
    from room_for_doubt.models.conditioned_trajectory import ConditionedPredictor

    root, noisy = source
    out = tmp_path / "pipeline"
    events = []

    class Images:
        def __init__(self, root, train, download):
            self.train = train
            events.append("train" if train else "test")
            if not train:
                assert (out / "artifacts/all_policies_before_test.json").exists()
                assert all(
                    (out / "heads" / f"{m}_42/metrics.json").exists()
                    for m in ["mean", "full"]
                )
            self.data = np.random.default_rng(6).integers(
                0, 255, (40 if train else 8, 32, 32, 3), dtype=np.uint8
            )

        @property
        def targets(self):
            if self.train:
                raise AssertionError("Clean training labels accessed")
            return np.arange(8) % 2

    monkeypatch.setattr(runner, "CIFAR100", Images)
    monkeypatch.setattr(runner, "load_observed_labels", lambda *args: noisy)
    monkeypatch.setattr(
        artifacts,
        "make_pretrained_classifier",
        lambda classes, weights: TinyBase(classes),
    )

    class TinyHead(ConditionedPredictor):
        def __init__(self, classes, outputs=50, pretrained=False):
            torch.nn.Module.__init__(self)
            self.features = torch.nn.Identity()
            self.class_emb = torch.nn.Embedding(classes, 32)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(544, 16), torch.nn.ReLU(), torch.nn.Linear(16, outputs)
            )

        def extract_features(self, images):
            return images.mean((1, 2, 3))[:, None].expand(-1, 512)

        def predict_from_features(self, features, classes):
            result = super().predict_from_features(features, classes)
            return result.expand(-1, 50) if result.shape[1] == 1 else result

    def encoder(classes, backbone=None):
        model = TinyHead(classes)
        if backbone:
            model.features.load_state_dict(torch.load(backbone, weights_only=True))
        return model.eval()

    monkeypatch.setattr(caching, "frozen_encoder", encoder)
    monkeypatch.setattr(meta_training, "ConditionedPredictor", TinyHead)
    monkeypatch.setattr(
        meta_training,
        "ScalarMarginPredictor",
        lambda classes, pretrained=False: TinyHead(classes, 1),
    )
    cfg = OmegaConf.load(ROOT / "configs/post_kaggle.yaml")
    cfg.source_run = str(root)
    cfg.stage = "prepare"
    cfg.device = "cpu"
    cfg.seeds = [42]
    cfg.epochs = 2
    cfg.batch_size = 16
    cfg.views = 2
    cfg.image_batch_size = 2
    cfg.temperatures = [0.1, 1.0]
    cfg.min_free_gb = 0
    torch.set_num_threads(4)
    first = runner.run_pipeline(cfg, out)
    assert first["test_evaluated"] is False and "test" not in events
    cfg.stage = "train"
    second = runner.run_pipeline(cfg, out)
    assert second["test_evaluated"] is False and "test" not in events
    cfg.stage = "evaluate"
    result = runner.run_pipeline(cfg, out)
    assert (
        result["status"] == "complete"
        and result["test_samples"] == 8
        and events.count("test") == 1
    )
    assert {r["method"] for r in result["results"]} == {
        "base",
        "tta",
        "uw_msp",
        "uw_mean",
        "uw_full",
    }
    with np.load(out / "artifacts/test_base_none.npz") as f:
        assert np.array_equal(f["ids"], np.arange(8))
        assert f["probabilities"].shape == (8, 100)
    frozen = json.loads((out / "artifacts/all_policies_before_test.json").read_text())
    assert len(frozen["policies"]) == 3
    val = np.load(out / "cache/val_probs.npy")
    test = np.load(out / "cache/test_probs.npy")
    assert val.shape == (4, 2, 100) and test.shape == (8, 2, 100)
    # Reusing a finished head and cache must not silently accept changed science.
    cfg.views = 3
    with pytest.raises(ValueError, match="scientific config differs"):
        runner.run_pipeline(cfg, out)
    cfg.views = 2
    path = root / "checkpoints/best.pt"
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["state_dict"]["fc.bias"][0] += 1
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="another source"):
        runner.run_pipeline(cfg, out)


def test_kaggle_launcher_finds_only_complete_source(source, tmp_path):
    import runpy

    find = runpy.run_path(str(ROOT / "scripts/kaggle_postprocess.py"))["find_source"]
    root, _ = source
    assert find(tmp_path) == root
    status = json.loads((root / "metrics.json").read_text())
    status["status"] = "paused_time_budget"
    (root / "metrics.json").write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match="Найдено: 0"):
        find(tmp_path)


def test_kaggle_launcher_rejects_ambiguous_inputs(source, tmp_path):
    import runpy
    import shutil

    find = runpy.run_path(str(ROOT / "scripts/kaggle_postprocess.py"))["find_source"]
    root, _ = source
    shutil.copytree(root, tmp_path / "another_source")
    with pytest.raises(RuntimeError, match="Найдено: 2"):
        find(tmp_path)
