import numpy as np
from room_for_doubt.evaluation.selection import (
    select_threshold,
    top2_correct,
    select_temperature,
    weighted_consensus,
)
from room_for_doubt.evaluation.campaign_cache import make_views, completed_rows
import pytest
from room_for_doubt.evaluation.selection import paired_accuracy_interval


def test_top2_gate_noop_wins_validation_tie():
    top = np.array([[0, 1], [1, 0]])
    b = np.array([[0.8, 0.1], [0.1, 0.8]])
    selected, _ = select_threshold(top, b, np.array([0, 1]), [1.0, 0.5])
    assert selected == 1.0
    pred, mask = top2_correct(top, b, selected)
    assert not mask.any()
    np.testing.assert_array_equal(pred, [0, 1])


def test_uniform_tta_wins_validation_tie_and_weighting_normalizes():
    p = np.array([[[0.8, 0.2], [0.7, 0.3]], [[0.1, 0.9], [0.2, 0.8]]])
    b = np.array([[0.0, 1.0], [1.0, 0.0]])
    selected, _ = select_temperature(p, b, np.array([0, 1]), [0.5, 2.0])
    assert selected == "uniform"
    result = weighted_consensus(p, b, 0.5)
    np.testing.assert_allclose(result.sum(1), 1.0)


def test_views_are_stable_and_keep_original_first():
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    a, b = make_views(image, 20, 42), make_views(image, 20, 42)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[0], image)
    assert np.asarray(a).shape == (20, 32, 32, 3)


def test_paired_interval_has_zero_width_for_identical_predictions():
    labels = np.array([0, 1, 1, 0])
    pred = np.array([0, 0, 1, 1])
    assert paired_accuracy_interval(labels, pred, pred) == dict(
        gain_pp=0.0, ci_low_pp=0.0, ci_high_pp=0.0
    )


def test_partial_cache_keeps_only_completed_prefix():
    p = np.zeros((4, 2, 3), dtype=np.float32)
    f = np.zeros((4, 2, 5), dtype=np.float32)
    p[:2] = 1 / 3
    f[:2] = 1
    assert completed_rows(p, f) == 2
    p[3] = 1 / 3
    f[3] = 1
    with pytest.raises(ValueError, match="prefix"):
        completed_rows(p, f)
