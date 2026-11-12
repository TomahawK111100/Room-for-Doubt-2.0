"""Freeze boundary and paired external-evaluation checks."""

import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn

from room_for_doubt.data import cifar10_1
from room_for_doubt.evaluation import confirmation
from room_for_doubt.evaluation.posthoc_runtime import ForwardPlan


def test_all_frozen_seeds_required_before_loading_data(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Test data opened before cohort validation")

    monkeypatch.setattr(confirmation, "load_v6", forbidden)
    with pytest.raises(ValueError, match="all three"):
        confirmation.evaluate({42: tmp_path}, tmp_path, tmp_path, tmp_path, "x", "cpu")

    def missing(*args, **kwargs):
        raise ValueError("Invalid S4 manifest")

    monkeypatch.setattr(confirmation, "verify_frozen", missing)
    with pytest.raises(ValueError, match="manifest"):
        confirmation.evaluate(
            {s: tmp_path for s in (42, 43, 44)},
            tmp_path,
            tmp_path,
            tmp_path,
            "x",
            "cpu",
        )


def test_release_checks_blob_and_rejects_overlap(tmp_path, monkeypatch):
    images = np.random.default_rng(3).integers(
        0, 256, (2000, 32, 32, 3), dtype=np.uint8
    )
    labels = np.repeat(np.arange(10), 200)
    assets = {}
    for name, array in (("images.npy", images), ("labels.npy", labels)):
        path = tmp_path / name
        np.save(path, array)
        value = path.read_bytes()
        assets[name] = (
            hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest(),
            len(value),
        )
    monkeypatch.setattr(cifar10_1, "ASSETS", assets)
    verified, y, _ = cifar10_1.load_v6(tmp_path, [])
    assert np.array_equal(verified, images) and np.array_equal(y, labels)
    with pytest.raises(ValueError, match="overlap"):
        cifar10_1.load_v6(tmp_path, images[:1])
    (tmp_path / "labels.npy").write_bytes(b"altered")
    with pytest.raises(ValueError, match="release"):
        cifar10_1.load_v6(tmp_path, [])


def test_stratified_resampling_shares_image_ids_across_bases():
    labels = np.repeat([0, 1], 4)
    diff = np.array([1, -1, 1, -1, 0, 1, 0, -1])
    assert confirmation.paired_stratified_interval([diff, -diff, diff * 0], labels) == [
        0,
        0,
    ]
    assert confirmation.paired_stratified_interval([labels, labels], labels) == [50, 50]


def test_direct_inference_uses_single_backbone_forward():
    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 2)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return self.fc(x)

    model, head = Base(), nn.Linear(3, 2)
    x = torch.ones(2, 3)
    plan = ForwardPlan({"b": model}, "b", "direct", head=head, alpha=1)
    torch.testing.assert_close(plan(x), head(x).softmax(1))
    assert model.calls == 1


def test_fit_notebooks_exclude_external_data_code(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "builder", "scripts/build_confirmation_s4.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    for seed in (42, 43, 44):
        builder.build(tmp_path, seed)
        manifest = json.loads((tmp_path / f"seed{seed}_manifest.json").read_text())
        assert manifest["seed"] == seed and manifest["stage"] == "fit"
        assert not any(
            p.endswith(("data/cifar10_1.py", "evaluation/confirmation.py"))
            for p in manifest["files"]
        )
