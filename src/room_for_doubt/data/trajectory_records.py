"""Check artifact identity and reuse the base-model split.

Clean label fields serve only as dataset identity checks here, not regression
targets. Historical clean-training-label detector supervision remains oracle.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from room_for_doubt.data.cifar_noisy import get_train_val_indices


def historical_trajectory_path(dataset_name, seed):
    """The CIFAR-100 `worse` basename is historical, not an annotation claim."""
    if dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return (
        Path("artifacts/trajectories")
        / f"stage1_trajectories_{dataset_name}n_worse_seed{seed}.json"
    )


def _clean_label(record):
    if "true_label" in record:
        return record["true_label"]
    if "clean_label" in record:
        return record["clean_label"]
    raise ValueError("Trajectory record lacks a clean-label identity field")


@dataclass
class TrajectoryRecords:
    records: dict
    train_ids: np.ndarray
    val_ids: np.ndarray

    def epoch_target(self, sample_id, length=50, start_epoch=0):
        """Return margins for explicit epochs, never observation positions."""
        if length < 1 or start_epoch < 0:
            raise ValueError("Epoch window requires length > 0 and start_epoch >= 0")
        if str(sample_id) not in self.records:
            raise ValueError(f"Missing trajectory target for sample {sample_id}")
        record = self.records[str(sample_id)]
        by_epoch = dict(zip(record["epochs"], record["margins"]))
        requested = range(start_epoch, start_epoch + length)
        missing = [epoch for epoch in requested if epoch not in by_epoch]
        if missing:
            raise MissingEpochError(f"Sample {sample_id} is missing epochs {missing}")
        return np.asarray([by_epoch[epoch] for epoch in requested], dtype=np.float32)

    def aligned_targets(self, length=50, start_epoch=0, missing_policy="drop"):
        """Select complete train cases without reassigning excluded IDs to val."""
        if missing_policy not in {"drop", "error"}:
            raise ValueError(f"Unknown missing-epoch policy: {missing_policy}")
        included, targets, excluded = [], [], []
        for sample_id in self.train_ids:
            try:
                target = self.epoch_target(int(sample_id), length, start_epoch)
            except MissingEpochError:
                if missing_policy == "error":
                    raise
                excluded.append(int(sample_id))
                continue
            included.append(int(sample_id))
            targets.append(target)
        if not included:
            raise ValueError(
                "No complete training trajectories in the requested epoch window"
            )
        return (
            np.asarray(included, dtype=np.int64),
            np.stack(targets),
            np.asarray(excluded, dtype=np.int64),
        )

    def oracle_detector_arrays(self, length=50, start_epoch=0, missing_policy="drop"):
        """Historical clean-training-label supervision, explicitly oracle."""
        included, targets, _ = self.aligned_targets(length, start_epoch, missing_policy)
        labels = [
            int(
                self.records[str(i)]["noisy_label"]
                != _clean_label(self.records[str(i)])
            )
            for i in included
        ]
        return targets, np.asarray(labels)


class MissingEpochError(ValueError):
    """An otherwise valid trajectory lacks a requested epoch."""


def load_verified_trajectories(
    path, dataset_name, clean_targets, seed=42, val_ratio=0.1
):
    """Fail before fitting if artifact identity or expected training IDs differ."""
    if dataset_name not in {"cifar10", "cifar100"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    with Path(path).open() as handle:
        records = json.load(handle)
    if not isinstance(records, dict) or not records:
        raise ValueError("Trajectory file must contain nonempty sample records")
    clean_targets = np.asarray(clean_targets)
    train_ids, val_ids = get_train_val_indices(len(clean_targets), val_ratio, seed)
    expected, actual = set(train_ids.tolist()), set()
    mismatches = 0
    classes = 10 if dataset_name == "cifar10" else 100
    for key, record in records.items():
        try:
            sample_id = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid sample ID: {key}") from exc
        if key != str(sample_id) or not 0 <= sample_id < len(clean_targets):
            raise ValueError(f"Invalid sample ID: {key}")
        actual.add(sample_id)
        if (
            not isinstance(record, dict)
            or "noisy_label" not in record
            or "margins" not in record
        ):
            raise ValueError(f"Sample {sample_id} lacks noisy_label or margins")
        noisy = record["noisy_label"]
        if (
            not isinstance(noisy, int)
            or isinstance(noisy, bool)
            or not 0 <= noisy < classes
        ):
            raise ValueError(
                f"Invalid noisy label for {dataset_name}, sample {sample_id}"
            )
        if _clean_label(record) != int(clean_targets[sample_id]):
            mismatches += 1
        margins = np.asarray(record["margins"], dtype=float)
        if margins.ndim != 1 or not margins.size or not np.isfinite(margins).all():
            raise ValueError(f"Sample {sample_id} has invalid or missing margins")
        epochs = record.get("epochs")
        if not isinstance(epochs, list) or len(epochs) != len(margins):
            raise ValueError(
                f"Sample {sample_id} requires epochs aligned one-to-one with margins"
            )
        if any(not isinstance(e, int) or isinstance(e, bool) or e < 0 for e in epochs):
            raise ValueError(f"Sample {sample_id} has invalid epoch numbers")
        if len(set(epochs)) != len(epochs):
            raise ValueError(f"Sample {sample_id} has duplicate epoch numbers")
    if mismatches:
        raise ValueError(
            f"Trajectory dataset mismatch: {mismatches}/{len(records)} clean-label identity fields disagree with {dataset_name}. Check the source file."
        )
    if actual != expected:
        raise ValueError(
            f"Trajectory split mismatch for seed {seed}: {len(expected - actual)} missing train IDs, {len(actual - expected)} unexpected IDs. Use the base-model split."
        )
    return TrajectoryRecords(records, train_ids, val_ids)
