import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from room_for_doubt.data import animal10n
from room_for_doubt.data.cifar_noisy import load_observed_labels, load_noisy_labels
from room_for_doubt.data.observed_images import imagenet_transform
from room_for_doubt.evaluation.campaign_cache import make_views

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setattr(animal10n, "EXPECTED_COUNTS", {True: 40, False: 8})
    for train, count, name in [
        (True, 40, "data_batch_1.bin"),
        (False, 8, "test_batch.bin"),
    ]:
        records = np.zeros(count, dtype=animal10n.RECORD)
        records["id"] = np.arange(count) + 100  # Raw IDs need not be offsets.
        records["label"] = np.arange(count) % 10
        records["pixels"] = np.random.default_rng(count).integers(
            0, 256, (count, 3, 64, 64), dtype=np.uint8
        )
        records.tofile(tmp_path / name)
    return tmp_path


def test_binary_pixels_labels_and_worker_pickle(release):
    raw = animal10n.Animal10N(release)
    stored = np.fromfile(release / "data_batch_1.bin", dtype=animal10n.RECORD)
    np.testing.assert_array_equal(raw.data[3], stored[3]["pixels"].transpose(1, 2, 0))
    assert raw.data[3].shape == (64, 64, 3)
    assert raw.clean_training_targets is None and raw.source_ids[0] == 100
    np.testing.assert_array_equal(
        load_observed_labels(release, "animal10n", "human"), raw.targets
    )
    restored = pickle.loads(pickle.dumps(raw.data))
    assert (
        isinstance(restored.records[0], np.memmap) and restored.records[0].mode == "r"
    )
    np.testing.assert_array_equal(restored[3], raw.data[3])
    with pytest.raises(ValueError, match="Unsupported"):
        load_noisy_labels(release, "animal10n", "human")
    with pytest.raises(ValueError, match="human"):
        load_observed_labels(release, "animal10n", "clean")


def test_manifest_detects_pixel_change_even_when_labels_match(release):
    first = animal10n.file_manifest(release)
    cfg = dict(
        data_dir=str(release),
        dataset=dict(
            name="animal10n", native_size=64, manifest_sha256=first["manifest_sha256"]
        ),
    )
    animal10n.verify_identity(animal10n.Animal10N(release), cfg)
    p = release / "data_batch_1.bin"
    b = bytearray(p.read_bytes())
    b[100] ^= 1
    p.write_bytes(b)
    with pytest.raises(ValueError, match="identity"):
        animal10n.verify_identity(animal10n.Animal10N(release), cfg)


def test_missing_truncated_wrong_count_labels_and_duplicate_ids(release):
    p = release / "data_batch_1.bin"
    good = p.read_bytes()
    p.write_bytes(good[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        animal10n.Animal10N(release)
    p.write_bytes(good[: -animal10n.RECORD.itemsize])
    with pytest.raises(ValueError, match="size"):
        animal10n.Animal10N(release)
    records = np.frombuffer(good, dtype=animal10n.RECORD).copy()
    records[0]["label"] = 10
    records.tofile(p)
    with pytest.raises(ValueError, match="0..9"):
        animal10n.Animal10N(release)
    records[0]["label"] = 0
    records[1]["id"] = records[0]["id"]
    records.tofile(p)
    with pytest.raises(ValueError, match="Duplicate"):
        animal10n.Animal10N(release)
    p.unlink()
    with pytest.raises(FileNotFoundError, match="official"):
        animal10n.Animal10N(release, download=True)


def test_cifar_views_exactly_preserved_and_animal_keeps_64px():
    image = np.random.default_rng(3).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    rng = np.random.default_rng(71)
    old = [image]
    for _ in range(9):
        v = image[:, ::-1] if rng.random() < 0.5 else image
        v = np.pad(v, ((4, 4), (4, 4), (0, 0)), mode="reflect")
        y, x = rng.integers(0, 9, size=2)
        old.append(v[y : y + 32, x : x + 32].copy())
    np.testing.assert_array_equal(make_views(image, 10, 71), old)
    big = np.tile(image, (2, 2, 1))
    views = make_views(big, 10, 71)
    assert all(v.shape == (64, 64, 3) for v in views)
    np.testing.assert_array_equal(views[:5], make_views(big, 5, 71))
    transform = imagenet_transform(224, True, 64)
    assert (
        transform.transforms[0].size == (64, 64)
        and transform.transforms[0].padding == 8
    )


def test_animal_base_resume_exports_observed_trajectories_before_test(
    release, monkeypatch
):
    import room_for_doubt.training.kaggle_base as base

    events = []

    class TrackedAnimal(animal10n.Animal10N):
        def __init__(self, root, train=True, download=False):
            events.append("train" if train else "test")
            super().__init__(root, train, download)

    monkeypatch.setattr(base, "Animal10N", TrackedAnimal)
    monkeypatch.setattr(
        base,
        "make_pretrained_classifier",
        lambda *a: torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 10)
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    cfg = OmegaConf.load(ROOT / "configs/animal10n_pilot.yaml").base
    cfg.data_dir = str(release)
    cfg.dataset.manifest_sha256 = animal10n.file_manifest(release)["manifest_sha256"]
    cfg.require_cuda = False
    cfg.expected_cuda_devices = None
    cfg.image_size = 8
    cfg.epochs = 2
    cfg.num_workers = 0
    cfg.batch_size = 16
    cfg.max_hours = -1
    out = release / "run"
    assert base.train(cfg, out)["status"] == "paused_time_budget"
    assert "test" not in events
    cfg.max_hours = 1
    assert base.train(cfg, out)["status"] == "complete"
    assert events.count("test") == 1
    with np.load(out / "artifacts/meta_targets.npz") as data:
        assert data["dataset_name"] == "animal10n" and data["targets"].shape == (36, 2)
        assert not set(data["base_train_ids"]) & set(data["base_val_ids"])
        np.testing.assert_array_equal(
            data["noisy_labels"], np.arange(40)[data["sample_ids"]] % 10
        )
    assert (out / "report.md").is_file()


def test_pilot_data_ambiguity_and_frozen_configs(release):
    spec = importlib.util.spec_from_file_location(
        "animal_pilot", ROOT / "scripts/animal10n_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.locate_data(release) == release
    cfg = OmegaConf.load(ROOT / "configs/animal10n_pilot.yaml")
    for name in ["base", "study", "paired", "augmentation"]:
        run = module.make_config(cfg, name, release / "out", release, 42, "abc")
        assert "max_hours" not in run or run.max_hours > 0
    assert (
        cfg.base.epochs == 50
        and cfg.base.batch_size == 128
        and cfg.base.expected_cuda_devices == 2
    )
    assert cfg.study.early_epochs == 5 and list(cfg.study.seeds) == [42, 43, 44]
    assert cfg.augmentation.augmentation_views == 5
    second = release / "second"
    second.mkdir()
    (second / "test_batch.bin").touch()
    (second / "data_batch_1.bin").touch()
    with pytest.raises(ValueError, match="found 2"):
        module.locate_data(release)


def test_animal_study_and_augmented_control_without_clean_train_labels(
    release, tmp_path, monkeypatch
):
    """Real 64px binary adapter, tiny networks: verify the whole new data route."""
    import csv
    import room_for_doubt.analysis.feature_study as study
    import room_for_doubt.analysis.augmentation_control as control
    import room_for_doubt.data.kaggle_artifacts as artifacts
    import room_for_doubt.inference.post_kaggle_cache as cache
    from room_for_doubt.data.cifar_noisy import get_train_val_indices
    from room_for_doubt.data.meta_splits import reserve_meta_splits
    from tests.test_post_kaggle import TinyBase

    base = tmp_path / "source"
    (base / "checkpoints").mkdir(parents=True)
    (base / "artifacts").mkdir()
    frozen = OmegaConf.load(ROOT / "configs/animal10n_pilot.yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(frozen.base))
    cfg.image_size = 8
    cfg.data_dir = str(release)
    cfg.dataset.manifest_sha256 = animal10n.file_manifest(release)["manifest_sha256"]
    OmegaConf.save(cfg, base / "config.yaml")
    torch.save(
        dict(
            state_dict=TinyBase(10).state_dict(),
            epoch=7,
            val_loss=0.2,
            config=OmegaConf.to_container(cfg),
        ),
        base / "checkpoints/best.pt",
    )
    (base / "metrics.json").write_text(
        json.dumps(
            dict(
                status="complete", completed_epochs=50, best_epoch=7, best_val_loss=0.2
            )
        )
    )
    with (base / "history.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "val_loss"])
        w.writeheader()
        w.writerows(dict(epoch=i, val_loss=0.2 if i == 7 else 0.5) for i in range(50))
    train, val = get_train_val_indices(40, 0.1, 42)
    observed = animal10n.Animal10N(release).targets
    np.savez(
        base / "artifacts/meta_targets.npz",
        sample_ids=train,
        base_train_ids=train,
        base_val_ids=val,
        noisy_labels=observed[train],
        targets=np.random.default_rng(1)
        .uniform(-1, 1, (len(train), 50))
        .astype("float32"),
        epochs=np.arange(50),
        excluded_ids=np.array([], dtype=np.int64),
        dataset_name=np.array("animal10n"),
        noisy_split=np.array("human"),
        split_seed=np.array(20261015),
        **reserve_meta_splits(train, 20261015),
    )
    np.savez(base / "artifacts/split_ids.npz", train_ids=train, val_ids=val)
    monkeypatch.setattr(
        artifacts,
        "make_pretrained_classifier",
        lambda classes, weights: TinyBase(classes),
    )

    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Identity()

        def extract_features(self, x):
            return x.mean((1, 2, 3))[:, None].expand(-1, 512)

    monkeypatch.setattr(cache, "frozen_encoder", lambda *a: TinyEncoder())
    monkeypatch.setattr(
        study,
        "noise_and_epoch_profile",
        lambda *a: pytest.fail("No clean train labels"),
    )
    cfg = OmegaConf.create(OmegaConf.to_container(frozen.study))
    cfg.source_run = str(base)
    cfg.data_dir = str(release)
    cfg.device = "cpu"
    cfg.epochs = 2
    cfg.feature_dim = 2
    cfg.batch_size = 16
    cfg.seeds = [42]
    cfg.view_counts = [1, 2]
    cfg.temperatures = [0.1, 1.0]
    cfg.min_free_gb = 0
    out = tmp_path / "study"
    result = study.run_study(cfg, out)
    assert result["status"] == "complete" and result["heads_trained_or_reused"] == 6
    assert result["train_noise_diagnostic"] == "unavailable_no_clean_training_labels"
    assert not (out / "tables/real_history_epoch_noise.csv").exists()
    aug = OmegaConf.create(OmegaConf.to_container(frozen.augmentation))
    aug.source_run = str(base)
    aug.study_run = str(out)
    aug.post_run = str(out / "shared")
    aug.data_dir = str(release)
    aug.device = "cpu"
    aug.min_free_gb = 0
    aug.auroc_bootstrap_draws = 10
    result = control.run_control(aug, tmp_path / "augmentation")
    assert result["status"] == "complete" and result["new_head_count"] == 1
    marker = json.loads(
        (tmp_path / "augmentation/augmented_target/complete.json").read_text()
    )
    assert marker["transform"].startswith("RandomCrop64_padding8_constant")


def test_cuda_sanity_fails_before_allocating_when_gpu_missing(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "animal_pilot", ROOT / "scripts/animal10n_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="two visible CUDA GPUs"):
        module.cuda_sanity(tmp_path)
    assert not list(tmp_path.iterdir())


def test_primary_interval_uses_selected_uniform_budget_and_declared_seed(
    tmp_path, monkeypatch
):
    import room_for_doubt.evaluation.feature_study as evaluation
    from room_for_doubt.evaluation.selection import paired_accuracy_interval

    (tmp_path / "artifacts").mkdir()
    (tmp_path / "tables").mkdir()
    p = np.array(
        [
            [[0.6, 0.4], [0.0, 1.0]],
            [[0.6, 0.4], [0.0, 1.0]],
            [[0.6, 0.4], [1.0, 0.0]],
            [[0.6, 0.4], [1.0, 0.0]],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    name = "base_resnet50__early_mean__42"
    policies = [
        dict(method=m, views=v, temperature="uniform")
        for m, v in [("tta", 1), ("tta", 2), ("msp", 1), ("msp", 2), (name, 2)]
    ]
    frozen = dict(
        policies=policies,
        selected=dict(tta=policies[0], msp=policies[2], **{name: policies[4]}),
    )
    (tmp_path / "artifacts/policies_before_test.json").write_text(json.dumps(frozen))
    score = 1 - p.max(-1)
    monkeypatch.setattr(
        evaluation, "score_heads", lambda *a: dict(msp=score, **{name: score})
    )
    seeds = []

    def interval(*a, **kw):
        seeds.append(kw.get("seed"))
        return paired_accuracy_interval(*a, **kw)

    monkeypatch.setattr(evaluation, "paired_accuracy_interval", interval)
    selected, aggregate = evaluation.evaluate_policies(
        OmegaConf.create(dict(bootstrap_seed=20261019)),
        tmp_path,
        p,
        labels,
        {},
        "cpu",
        frozen,
    )
    actual = next(r for r in selected if r["method"] == name)
    assert actual["gain_pp_vs_tta"] == 0  # Existing same-budget comparison retained.
    assert actual["gain_pp_vs_tta_selected"] == -50
    assert (
        actual["selected_tta_reference_views"] == 1
        and actual["tta_reference_views"] == 2
    )
    assert seeds and set(seeds) == {20261019}
    assert (
        next(r for r in aggregate if r["method"] == "base_resnet50__early_mean")[
            "gain_vs_selected_tta_pp"
        ]
        == -50
    )


def test_partial_pilot_records_previous_test_use(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "animal_pilot", ROOT / "scripts/animal10n_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert not module.test_evaluation_seen(tmp_path)
    (tmp_path / "base").mkdir()
    (tmp_path / "feature_study").mkdir()
    (tmp_path / "base/metrics.json").write_text(
        json.dumps(dict(status="complete", test_acc=0.8))
    )
    (tmp_path / "feature_study/metrics.json").write_text(
        json.dumps(dict(status="paused_time_budget", test_evaluated=False))
    )
    assert module.test_evaluation_seen(tmp_path)


def test_official_big_endian_records_independent_of_adapter_dtype(
    tmp_path, monkeypatch
):
    import struct

    monkeypatch.setattr(animal10n, "EXPECTED_COUNTS", {True: 2, False: 1})
    # Independent wire encoding from the author reader; do not use RECORD to write it.
    pixels = np.arange(3 * 64 * 64, dtype=np.uint8).reshape(3, 64, 64)
    payload = struct.pack(">ii", 123, 1) + pixels.tobytes()
    (tmp_path / "data_batch_1.bin").write_bytes(
        payload + struct.pack(">ii", 456, 9) + pixels.tobytes()
    )
    (tmp_path / "test_batch.bin").write_bytes(
        struct.pack(">ii", 0, 2) + pixels.tobytes()
    )
    raw = animal10n.Animal10N(tmp_path)
    np.testing.assert_array_equal(raw.targets, [1, 9])
    np.testing.assert_array_equal(raw.source_ids, [123, 456])
    np.testing.assert_array_equal(raw.data[0], pixels.transpose(1, 2, 0))
    assert animal10n.file_manifest(tmp_path)["format"].endswith("_be_v1")
