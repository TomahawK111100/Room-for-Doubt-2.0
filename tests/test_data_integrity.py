"""Offline checks for real-label loading and import-time integrity behavior."""

import importlib
import numpy as np
import pytest
import torch
from torchvision.datasets import CIFAR10, CIFAR100

from room_for_doubt.data import cifar_noisy
from room_for_doubt.utils import reproducibility


def test_missing_annotations_never_become_random_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(cifar_noisy, "_download", lambda *args: None)
    with pytest.raises(FileNotFoundError, match="Official CIFAR-N"):
        cifar_noisy.load_noisy_labels(str(tmp_path), "cifar10n", "worse")


def test_failed_download_removes_partial_file(tmp_path, monkeypatch):
    target = tmp_path / "CIFAR-10_human.pt"

    def fail(url, path):
        target.write_bytes(b"incomplete")
        raise OSError("offline")

    monkeypatch.setattr(cifar_noisy.urllib.request, "urlretrieve", fail)
    with pytest.raises(RuntimeError, match="official annotation"):
        cifar_noisy.load_noisy_labels(str(tmp_path), "cifar10n", "worse")
    assert not target.exists()


def test_annotation_shape_validation(tmp_path):
    torch.save(
        {"worse_label": [1, 2], "clean_label": [1, 2]}, tmp_path / "CIFAR-10_human.pt"
    )
    with pytest.raises(ValueError, match="50000"):
        cifar_noisy.load_noisy_labels(str(tmp_path), "cifar10n", "worse")


def test_official_label_arrays_are_preserved(tmp_path):
    clean = np.arange(50000) % 10
    noisy = (clean + 1) % 10
    torch.save(
        {"worse_label": noisy, "clean_label": clean}, tmp_path / "CIFAR-10_human.pt"
    )
    actual_noisy, actual_clean = cifar_noisy.load_noisy_labels(
        str(tmp_path), "cifar10n", "worse"
    )
    np.testing.assert_array_equal(actual_noisy, noisy)
    np.testing.assert_array_equal(actual_clean, clean)


def test_seed_module_preserves_torchvision_integrity_methods():
    methods = (CIFAR10._check_integrity, CIFAR100._check_integrity)
    importlib.reload(reproducibility)
    assert methods == (CIFAR10._check_integrity, CIFAR100._check_integrity)


def test_split_is_disjoint_and_complete():
    train, val = cifar_noisy.get_train_val_indices(100, 0.1, 42)
    assert len(train) == 90 and len(val) == 10
    assert not set(train) & set(val)
    np.testing.assert_array_equal(np.sort(np.concatenate([train, val])), np.arange(100))
