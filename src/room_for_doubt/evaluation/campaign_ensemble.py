"""Predeclared 2/3-model baselines; no test-based ensemble member selection."""

from pathlib import Path
import numpy as np
from room_for_doubt.evaluation.diagnostic_metrics import (
    probability_metrics,
    detection_metrics,
)
from room_for_doubt.evaluation.selection import paired_accuracy_interval
from room_for_doubt.reporting.artifacts import write_csv


def evaluate_ensembles(source_post, completed_jobs, out):
    out = Path(out)
    reference = Path(source_post) / "cache"
    ids = np.load(reference / "test_ids.npy")
    labels = np.load(reference / "test_labels.npy")
    base = np.load(reference / "test_probs.npy", mmap_mode="r")[:, 0]
    members = {42: base}
    for job in completed_jobs:
        if job["dataset"] != "cifar100n":
            continue
        cache = Path(job["study_run"]) / "shared/cache"
        if not np.array_equal(
            np.load(cache / "test_ids.npy"), ids
        ) or not np.array_equal(np.load(cache / "test_labels.npy"), labels):
            raise ValueError("Ensemble test IDs/labels differ")
        members[job["seed"]] = np.load(cache / "test_probs.npy", mmap_mode="r")[:, 0]
    rows = []
    for seeds in ([42], [42, 43], [42, 43, 44]):
        if not all(seed in members for seed in seeds):
            continue
        p = np.mean([members[seed] for seed in seeds], axis=0)
        score = 1 - p.max(1)
        own = detection_metrics(p.argmax(1) != labels, score)
        common = detection_metrics(base.argmax(1) != labels, score)
        interval = paired_accuracy_interval(labels, base.argmax(1), p.argmax(1))
        rows.append(
            dict(
                base_seeds="/".join(map(str, seeds)),
                models=len(seeds),
                **probability_metrics(labels, p),
                own_error_auroc=own["auroc"],
                base42_error_auroc=common["auroc"],
                **interval,
            )
        )
        np.savez_compressed(
            out / "artifacts" / f"ensemble_{len(seeds)}.npz",
            ids=ids,
            labels=labels,
            probabilities=p,
        )
    write_csv(out / "tables/ensemble_baselines.csv", rows)
    return rows
