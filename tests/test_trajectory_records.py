"""Tiny synthetic records are test fixtures, not research ground truth."""

import json
import numpy as np
import pytest

from room_for_doubt.data.cifar_noisy import get_train_val_indices
from room_for_doubt.data.trajectory_records import (
    historical_trajectory_path,
    load_verified_trajectories,
)
from room_for_doubt.data import conditioned_trajectories


@pytest.fixture
def fixture_records(tmp_path):
    clean = np.arange(20) % 10
    train, val = get_train_val_indices(20, 0.1, 42)
    records = {
        str(i): {
            "noisy_label": int(clean[i]),
            "true_label": int(clean[i]),
            "margins": [float(i)] * 50,
            "epochs": list(range(50)),
        }
        for i in train
    }
    path = tmp_path / "trajectories.json"
    path.write_text(json.dumps(records))
    return path, clean, train, val, records


def test_split_matches_base_and_all_requests_have_targets(fixture_records):
    path, clean, train, val, records = fixture_records
    verified = load_verified_trajectories(path, "cifar10", clean)
    np.testing.assert_array_equal(verified.train_ids, train)
    np.testing.assert_array_equal(verified.val_ids, val)
    assert set(train).isdisjoint(val)
    assert set(train) == {int(i) for i in records}
    assert len(verified.oracle_detector_arrays()[0]) == len(train)


def test_rejects_wrong_dataset_identity(fixture_records):
    path, clean, *_ = fixture_records
    with pytest.raises(ValueError, match="dataset mismatch"):
        load_verified_trajectories(path, "cifar100", clean + 10)


def test_rejects_a_missing_train_target(fixture_records):
    path, clean, train, _, records = fixture_records
    del records[str(train[0])]
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="1 missing train IDs"):
        load_verified_trajectories(path, "cifar10", clean)


def test_rejects_validation_samples_in_training_file(fixture_records):
    path, clean, _, val, records = fixture_records
    i = int(val[0])
    records[str(i)] = {
        "noisy_label": int(clean[i]),
        "true_label": int(clean[i]),
        "margins": [1.0] * 50,
        "epochs": list(range(50)),
    }
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="1 unexpected IDs"):
        load_verified_trajectories(path, "cifar10", clean)


def test_short_targets_and_missing_ids_never_become_zeros(fixture_records):
    path, clean, train, val, records = fixture_records
    records[str(train[0])]["margins"] = [0.2] * 49
    records[str(train[0])]["epochs"] = list(range(49))
    path.write_text(json.dumps(records))
    verified = load_verified_trajectories(path, "cifar10", clean)
    with pytest.raises(ValueError, match="missing epochs"):
        verified.epoch_target(int(train[0]))
    with pytest.raises(ValueError, match="Missing trajectory target"):
        verified.epoch_target(int(val[0]))


def test_missing_labels_and_nonfinite_margins_fail(fixture_records):
    path, clean, train, _, records = fixture_records
    i = str(train[0])
    records[i].pop("noisy_label")
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="lacks noisy_label"):
        load_verified_trajectories(path, "cifar10", clean)
    records[i]["noisy_label"] = int(clean[train[0]])
    records[i]["margins"][0] = float("nan")
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="invalid or missing margins"):
        load_verified_trajectories(path, "cifar10", clean)


def test_dataset_indexes_images_by_stable_sample_id(fixture_records, monkeypatch):
    path, clean, train, _, _ = fixture_records

    class FakeCIFAR:
        def __init__(self, **kwargs):
            self.targets = clean

        def __getitem__(self, i):
            return f"image-{i}", int(clean[i])

    monkeypatch.setattr(
        conditioned_trajectories.torchvision.datasets, "CIFAR10", FakeCIFAR
    )
    dataset = conditioned_trajectories.CondTrajDataset(traj_file=path)
    assert len(dataset) == 18
    image, label, margin = dataset[0]
    assert image == f"image-{train[0]}"
    assert label.item() == clean[train[0]]
    assert margin.shape == (50,) and margin[0].item() == train[0]


def test_flow_default_avoids_mislabeled_cifar100_alias():
    path = historical_trajectory_path("cifar100", 42)
    assert path.name == "stage1_trajectories_cifar100n_worse_seed42.json"


@pytest.mark.parametrize("ratio", [-0.1, 0, 1, 1.1, 0.001])
def test_invalid_or_empty_splits_fail(ratio):
    with pytest.raises(ValueError, match="Split"):
        get_train_val_indices(20, ratio, 42)


def test_alignment_uses_epoch_numbers_not_list_positions(fixture_records):
    path, clean, train, _, records = fixture_records
    i = str(train[0])
    records[i]["epochs"] = list(reversed(range(60)))
    records[i]["margins"] = [float(e) for e in records[i]["epochs"]]
    path.write_text(json.dumps(records))
    verified = load_verified_trajectories(path, "cifar10", clean)
    np.testing.assert_array_equal(verified.epoch_target(int(i)), np.arange(50))
    np.testing.assert_array_equal(
        verified.epoch_target(int(i), length=10, start_epoch=50), np.arange(50, 60)
    )


def test_later_observations_do_not_replace_an_early_missing_epoch(fixture_records):
    path, clean, train, _, records = fixture_records
    i = str(train[0])
    records[i]["epochs"] = [e for e in range(60) if e != 7]
    records[i]["margins"] = [float(e) for e in records[i]["epochs"]]
    path.write_text(json.dumps(records))
    verified = load_verified_trajectories(path, "cifar10", clean)
    with pytest.raises(ValueError, match=r"missing epochs \[7\]"):
        verified.epoch_target(int(i))
    with pytest.raises(ValueError, match="missing epochs"):
        verified.aligned_targets(missing_policy="error")
    ids, targets, excluded = verified.aligned_targets(missing_policy="drop")
    assert int(i) not in ids and excluded.tolist() == [int(i)]
    assert targets.shape == (17, 50)
    assert set(ids) | set(excluded) == set(train)
    assert not set(excluded) & set(verified.val_ids)
    assert verified.oracle_detector_arrays()[0].shape == targets.shape


@pytest.mark.parametrize(
    "bad_epochs",
    [
        None,
        list(range(49)),
        [0] * 50,
        [True] + list(range(1, 50)),
        [-1] + list(range(1, 50)),
        [0.5] + list(range(1, 50)),
    ],
)
def test_invalid_epoch_metadata_is_rejected(fixture_records, bad_epochs):
    path, clean, train, _, records = fixture_records
    records[str(train[0])]["epochs"] = bad_epochs
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="epoch"):
        load_verified_trajectories(path, "cifar10", clean)


def test_all_incomplete_cases_fail_without_fabricated_targets(fixture_records):
    path, clean, _, _, records = fixture_records
    for record in records.values():
        record["epochs"] = list(range(1, 51))
    path.write_text(json.dumps(records))
    verified = load_verified_trajectories(path, "cifar10", clean)
    with pytest.raises(ValueError, match="No complete training trajectories"):
        verified.aligned_targets()
