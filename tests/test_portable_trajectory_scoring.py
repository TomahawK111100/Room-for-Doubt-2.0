import numpy as np
import pytest
import torch
from room_for_doubt.evaluation.trajectory_scoring import (
    mean_margin_badness,
    quantile_thresholds,
)
from room_for_doubt.data import cifar_noisy


def test_mean_margin_has_fixed_scale_without_labels():
    paths = np.array([[1.0, 1.0], [-1.0, -1.0], [0.5, -0.5], [2.0, 2.0]])
    np.testing.assert_allclose(mean_margin_badness(paths), [0.0, 1.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="finite"):
        mean_margin_badness(np.array([[np.nan]]))


def test_quantile_candidates_always_include_noop():
    scores = np.array([[0.42, 0.47], [0.43, 0.46]])
    actual = quantile_thresholds(scores, [0.25, 0.5, 0.75])
    assert actual[-1] == 1.0
    np.testing.assert_allclose(
        actual[:-1], np.quantile(scores.reshape(-1), [0.25, 0.5, 0.75])
    )


def test_observed_annotations_do_not_require_clean_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(cifar_noisy, "_download", lambda *args: None)
    torch.save(
        {"worse_label": np.ones(50000, dtype=np.int64)}, tmp_path / "CIFAR-10_human.pt"
    )
    actual = cifar_noisy.load_observed_labels(str(tmp_path), "cifar10n", "worse")
    np.testing.assert_array_equal(actual, np.ones(50000, dtype=np.int64))


def test_trajectory_training_loader_keeps_last_partial_batch():
    dm = cifar_noisy.CIFARDataModule(batch_size=4, num_workers=0)
    values = torch.arange(11)
    observed = torch.cat(list(dm._loader(values, shuffle=True)))
    torch.testing.assert_close(observed.sort().values, values)
