import itertools
import numpy as np
import pytest
from room_for_doubt.evaluation.tie_aware_risk import expected_tie_aurc


def test_tied_group_matches_average_of_all_tie_orderings():
    conf = np.array([0.9, 0.8, 0.8, 0.8, 0.2])
    errors = np.array([0, 1, 0, 1, 1])
    brute = []
    for order in itertools.permutations([1, 2, 3]):
        e = errors[[0, *order, 4]]
        brute.append(np.mean(np.cumsum(e) / np.arange(1, 6)))
    assert np.isclose(expected_tie_aurc(conf, errors), np.mean(brute))
    p = np.array([3, 4, 1, 0, 2])
    assert expected_tie_aurc(conf[p], errors[p]) == expected_tie_aurc(conf, errors)


def test_all_tied_and_distinct_confidences():
    assert np.isclose(expected_tie_aurc(np.ones(4), [0, 0, 1, 1]), 0.5)
    assert np.isclose(
        expected_tie_aurc([0.9, 0.8, 0.2], [0, 1, 1]), (0 + 0.5 + 2 / 3) / 3
    )
    with pytest.raises(ValueError):
        expected_tie_aurc([], [])
    with pytest.raises(ValueError):
        expected_tie_aurc([0.2, 0.3], [0, 2])
