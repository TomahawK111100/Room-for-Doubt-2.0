"""Reserve meta-validation and detector sets without consulting labels."""

import numpy as np


def reserve_meta_splits(sample_ids, seed):
    ids = np.asarray(sample_ids, dtype=np.int64).copy()
    if ids.ndim != 1 or len(ids) < 20 or len(np.unique(ids)) != len(ids):
        raise ValueError("Require at least 20 unique sample IDs")
    np.random.default_rng(seed).shuffle(ids)
    a, b, c = int(len(ids) * 0.6), int(len(ids) * 0.7), int(len(ids) * 0.85)
    return dict(
        meta_train_ids=ids[:a],
        meta_val_ids=ids[a:b],
        detector_fit_ids=ids[b:c],
        detector_eval_ids=ids[c:],
    )


def validate_meta_splits(splits, sample_ids):
    ids = np.asarray(sample_ids)
    if (
        ids.ndim != 1
        or not np.issubdtype(ids.dtype, np.integer)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Sample IDs must be unique one-dimensional integers")
    expected = set(np.asarray(sample_ids).tolist())
    seen = set()
    for name in (
        "meta_train_ids",
        "meta_val_ids",
        "detector_fit_ids",
        "detector_eval_ids",
    ):
        part = np.asarray(splits[name])
        if part.ndim != 1 or not np.issubdtype(part.dtype, np.integer):
            raise ValueError("Partition IDs must be one-dimensional integers")
        values = part.tolist()
        if not values or len(set(values)) != len(values) or seen & set(values):
            raise ValueError(
                "Meta and detector partitions must be nonempty, unique and disjoint"
            )
        seen.update(values)
    if seen != expected:
        raise ValueError("Partitions must cover exactly the usable sample IDs")
