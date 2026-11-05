"""Strict, hash-bound cache interface; no clean training labels or test arrays."""

import json
from pathlib import Path
import numpy as np
from room_for_doubt.data.meta_splits import reserve_meta_splits
from room_for_doubt.training.temporal_snapshots import file_digest


def finite(array):
    return all(
        np.isfinite(array[s : s + 1024]).all() for s in range(0, len(array), 1024)
    )


def probabilities(array):
    if array.ndim != 2 or not finite(array):
        raise ValueError("Require finite class-probability matrix")
    for start in range(0, len(array), 1024):
        part = array[start : start + 1024]
        if np.any(part < 0) or not np.allclose(part.sum(1), 1, atol=1e-5):
            raise ValueError("Invalid class probabilities")


def load_evidence(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    required = {
        "train_ids",
        "val_ids",
        "train_features",
        "train_probs",
        "teacher_probs",
        "val_features",
        "val_probs",
        "val_teacher_probs",
        "val_noisy_labels",
        "meta_train_rows",
        "meta_val_rows",
    }
    if (
        manifest.get("schema") != 1
        or manifest.get("scope") != "temporal_residual_pilot"
        or set(manifest["arrays"]) != required
    ):
        raise ValueError("Unsupported temporal evidence schema/scope/arrays")
    arrays = {}
    for name in sorted(required):
        item = manifest["arrays"][name]
        if (
            item["file"] != f"{name}.npy"
            or file_digest(root / item["file"]) != item["sha256"]
        ):
            raise ValueError(f"Evidence SHA256/name mismatch: {name}")
        arrays[name] = np.load(root / item["file"], mmap_mode="r", allow_pickle=False)
    classes, width = manifest["classes"], manifest["feature_width"]
    if manifest["class_axis"] != list(range(classes)) or classes < 2 or width < 1:
        raise ValueError("Invalid evidence class axis/feature width")
    for split in ("train", "val"):
        ids, f, p = (arrays[f"{split}_{key}"] for key in ("ids", "features", "probs"))
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids)
            or len(ids) == 0
            or np.any(ids < 0)
            or f.shape != (len(ids), width)
            or f.dtype != np.float32
            or not finite(f)
            or p.shape != (len(ids), classes)
        ):
            raise ValueError("Evidence IDs/features/class dimensions differ")
        probabilities(p)
    if np.intersect1d(arrays["train_ids"], arrays["val_ids"]).size:
        raise ValueError("Base train and validation IDs overlap")
    teacher = arrays["teacher_probs"]
    if teacher.shape != arrays["train_probs"].shape:
        raise ValueError("Teacher/student ID or class dimensions differ")
    probabilities(teacher)
    if arrays["val_teacher_probs"].shape != arrays["val_probs"].shape:
        raise ValueError("Validation teacher class dimensions differ")
    probabilities(arrays["val_teacher_probs"])
    y = arrays["val_noisy_labels"]
    if (
        y.shape != arrays["val_ids"].shape
        or y.dtype.kind not in "iu"
        or np.any((y < 0) | (y >= classes))
    ):
        raise ValueError("Invalid noisy-validation labels")
    partitions = reserve_meta_splits(arrays["train_ids"], manifest["split_seed"])
    for name in ("train", "val"):
        rows = arrays[f"meta_{name}_rows"]
        if (
            rows.ndim != 1
            or rows.dtype.kind not in "iu"
            or np.any(rows < 0)
            or np.any(rows >= len(teacher))
            or not np.array_equal(
                arrays["train_ids"][rows], partitions[f"meta_{name}_ids"]
            )
        ):
            raise ValueError("Meta partition differs from frozen original protocol")
    return manifest, arrays


def seal_evidence(directory, metadata):
    root = Path(directory)
    names = (
        "train_ids",
        "val_ids",
        "train_features",
        "train_probs",
        "teacher_probs",
        "val_features",
        "val_probs",
        "val_teacher_probs",
        "val_noisy_labels",
        "meta_train_rows",
        "meta_val_rows",
    )
    manifest = dict(
        schema=1,
        scope="temporal_residual_pilot",
        **metadata,
        arrays={
            name: dict(file=f"{name}.npy", sha256=file_digest(root / f"{name}.npy"))
            for name in names
        },
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    load_evidence(root)
    return manifest
