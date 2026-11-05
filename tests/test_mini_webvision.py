"""File-list identity, label boundaries and the real transfer training path."""

import importlib.util
import pickle
import contextlib
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from omegaconf import OmegaConf

from room_for_doubt.data import mini_webvision as web
from room_for_doubt.data.cifar_noisy import load_observed_labels, load_noisy_labels
from room_for_doubt.data.observed_images import imagenet_transform
from room_for_doubt.evaluation.campaign_cache import make_views

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "TRAIN_ROWS", 150)
    monkeypatch.setattr(web, "EVALUATION_PER_CLASS", 1)
    (tmp_path / "info").mkdir()
    (tmp_path / "google").mkdir()
    (tmp_path / "val_images_256").mkdir()
    for train, count in [(True, 150), (False, 50)]:
        rows = []
        for i in range(count):
            name = f"google/{i:04}.jpg" if train else f"{i:04}.jpg"
            p = tmp_path / name if train else tmp_path / "val_images_256" / name
            Image.new("RGB", (31 + i % 3, 22), (i % 255, 30, 50)).save(p)
            rows.append(f"{name} {i % 50}\n")
        # Unselected classes may be absent; no arbitrary first50 folder mapping.
        rows.append("google/unselected.jpg 50\n")
        (tmp_path / "info" / web.LISTS[train]).write_text("".join(rows))
    return tmp_path


def test_original_order_observed_labels_and_lazy_pickle(release):
    raw = web.MiniWebVision(release)
    assert raw.source_paths[7] == "google/0007.jpg" and raw.source_ids[7] == 7
    assert raw.targets[7] == 7 and len(raw) == 150
    assert raw.clean_training_targets is None
    assert raw.data[0].shape == (256, 256, 3)
    assert np.array_equal(pickle.loads(pickle.dumps(raw.data))[0], raw.data[0])
    assert np.array_equal(
        load_observed_labels(release, "miniwebvision", "google_first50"), raw.targets
    )
    with pytest.raises(ValueError, match="Unsupported"):
        load_noisy_labels(release, "miniwebvision", "google_first50")
    with pytest.raises(ValueError, match="Google"):
        load_observed_labels(release, "miniwebvision", "clean")
    assert web.locate_data(release) == release


def test_missing_selected_image_is_not_silently_dropped(release):
    (release / "google/0007.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="Missing listed"):
        web.MiniWebVision(release)


def test_duplicate_and_remapped_lists_rejected(release):
    p = release / "info" / web.LISTS[True]
    original = p.read_text()
    p.write_text(original + "google/0000.jpg 0\n")
    with pytest.raises(ValueError, match="duplicate"):
        web.MiniWebVision(release)
    p.write_text(original.replace("google/0000.jpg", "train/0000.jpg"))
    with pytest.raises(ValueError, match="original Google"):
        web.MiniWebVision(release)


def test_manifest_binds_images_and_full_evaluation_membership(release):
    first = web.file_manifest(release)
    cfg = dict(
        data_dir=str(release),
        dataset=dict(
            name="miniwebvision",
            num_classes=50,
            native_size=256,
            noisy_split="google_first50",
            manifest_sha256=first["manifest_sha256"],
        ),
    )
    web.verify_identity(web.MiniWebVision(release), cfg)
    Image.new("RGB", (20, 20), (255, 0, 0)).save(release / "google/0000.jpg")
    with pytest.raises(ValueError, match="identity"):
        web.verify_identity(web.MiniWebVision(release), cfg)
    (release / "info/val_filelist.txt").write_text("0000.jpg 0\n")
    with pytest.raises(ValueError, match="every original class"):
        web.MiniWebVision(release, train=False)


def test_training_and_shared_views_preserve256_geometry():
    image = np.random.default_rng(2).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    assert imagenet_transform(16, True, 256)(Image.fromarray(image)).shape == (
        3,
        16,
        16,
    )
    a, b = make_views(image, 3, 4), make_views(image, 3, 4)
    assert all(v.shape == image.shape for v in a)
    assert all(np.array_equal(v, w) for v, w in zip(a, b))
    assert np.array_equal(a[0], image)


def test_real_base_training_resume_and_observed_trajectory(release, monkeypatch):
    import room_for_doubt.training.kaggle_base as base
    from room_for_doubt.data.meta_splits import validate_meta_splits

    seen = []

    class Tracked(web.MiniWebVision):
        def __init__(self, root, train=True, download=False):
            seen.append("train" if train else "evaluation")
            super().__init__(root, train, download)

    monkeypatch.setattr(base, "MiniWebVision", Tracked)
    monkeypatch.setattr(
        base,
        "make_pretrained_classifier",
        lambda *a: torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 50)
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = OmegaConf.load(ROOT / "configs/mini_webvision_pilot.yaml").base
    cfg.data_dir = str(release)
    cfg.dataset.manifest_sha256 = web.file_manifest(release)["manifest_sha256"]
    cfg.require_cuda, cfg.expected_cuda_devices = False, None
    cfg.image_size, cfg.epochs, cfg.batch_size, cfg.num_workers = 8, 2, 32, 0
    cfg.max_hours = -1
    out = release / "run"
    assert base.train(cfg, out)["status"] == "paused_time_budget"
    assert "evaluation" not in seen
    cfg.max_hours = 1
    result = base.train(cfg, out)
    assert result["status"] == "complete" and result["trajectory_samples"] == 135
    assert "target_70_reached" not in result
    with np.load(out / "artifacts/meta_targets.npz") as archive:
        data = {k: archive[k] for k in archive.files}
    assert data["targets"].shape == (135, 2)
    assert np.array_equal(
        data["noisy_labels"], web.MiniWebVision(release).targets[data["sample_ids"]]
    )
    validate_meta_splits(data, data["sample_ids"])
    best = torch.load(out / "checkpoints/best.pt", weights_only=True)
    assert best["epoch"] == result["best_epoch"]
    assert (out / "report.md").is_file()


def test_shared_launcher_makes_mini_config():
    spec = importlib.util.spec_from_file_location(
        "shared_transfer", ROOT / "scripts/animal10n_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = OmegaConf.load(ROOT / "configs/mini_webvision_pilot.yaml")
    base = module.make_config(
        cfg, "base", Path("/output"), Path("/data"), 42, "dataset-id"
    )
    assert base.run_id == "miniwebvision_seed42"
    assert base.dataset.manifest_sha256 == "dataset-id"
    assert base.dataset.num_classes == 50 and base.epochs == 50


def test_downloader_retains_only_exact_members_and_rejects_missing(tmp_path):
    from room_for_doubt.data.webvision_download import extract_selected

    image = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(image, format="JPEG")
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        for name in ["google/other.jpg", "./google/wanted.jpg", "unrequested.txt"]:
            info = tarfile.TarInfo(name)
            payload = image.getvalue()
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    with tarfile.open(fileobj=io.BytesIO(data.getvalue()), mode="r|*") as archive:
        assert (
            extract_selected(archive, {"google/wanted.jpg": tmp_path / "wanted.jpg"})
            == 1
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["wanted.jpg"]
    with tarfile.open(fileobj=io.BytesIO(data.getvalue()), mode="r|*") as archive:
        with pytest.raises(ValueError, match="did not contain"):
            extract_selected(archive, {"missing.jpg": tmp_path / "missing.jpg"})


def test_downloader_terms_and_output_boundary_before_network(tmp_path, monkeypatch):
    import room_for_doubt.data.webvision_download as downloader

    monkeypatch.setattr(
        downloader,
        "fetch_lists",
        lambda *a: pytest.fail("Network before permission/output check"),
    )
    with pytest.raises(PermissionError, match="explicitly accept"):
        downloader.fetch_images(tmp_path)
    with pytest.raises(ValueError, match="never store"):
        downloader.fetch_images("/kaggle/working/images", accept_terms=True)


def test_metadata_only_download_preserves_original_filelists(
    release, tmp_path, monkeypatch
):
    import room_for_doubt.data.webvision_download as downloader

    data = io.BytesIO()
    originals = {
        "info/" + name: (release / "info" / name).read_bytes()
        for name in web.LISTS.values()
    }
    with tarfile.open(fileobj=data, mode="w") as archive:
        for name, payload in originals.items():
            info = tarfile.TarInfo("./" + name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    requested = []

    @contextlib.contextmanager
    def metadata(url):
        requested.append(url)
        with tarfile.open(fileobj=io.BytesIO(data.getvalue()), mode="r|*") as stream:
            yield stream

    monkeypatch.setattr(downloader, "archive", metadata)
    out = tmp_path / "metadata_only"
    result = downloader.fetch_lists(out)
    assert result["training_rows"] == 150 and result["evaluation_rows"] == 50
    assert result["image_data_downloaded"] is False
    assert requested == [downloader.BASE_URL + "info.tar"]
    assert all(
        (out / name).read_bytes() == payload for name, payload in originals.items()
    )


def test_mini_study_paired_and_control_with_observed_only_labels(
    release, tmp_path, monkeypatch
):
    """Original file-list adapter and tiny models: exercise the full50-class route."""
    import csv
    import json
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
    frozen = OmegaConf.load(ROOT / "configs/mini_webvision_pilot.yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(frozen.base))
    cfg.image_size = 8
    cfg.data_dir = str(release)
    cfg.dataset.manifest_sha256 = web.file_manifest(release)["manifest_sha256"]
    OmegaConf.save(cfg, base / "config.yaml")
    torch.save(
        dict(
            state_dict=TinyBase(50).state_dict(),
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
    train, val = get_train_val_indices(150, 0.1, 42)
    observed = web.MiniWebVision(release).targets
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
        dataset_name=np.array("miniwebvision"),
        noisy_split=np.array("google_first50"),
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
    from room_for_doubt.analysis.feature_contrasts import run_contrasts

    paired = OmegaConf.create(OmegaConf.to_container(frozen.paired))
    paired.study_run, paired.post_run = str(out), str(out / "shared")
    paired.auroc_bootstrap_draws = 10
    assert run_contrasts(paired, tmp_path / "paired")["status"] == "complete"
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
    assert marker["transform"].startswith("RandomCrop256_padding32_constant")
