"""Paired conditional intervals for the pre-specified target comparisons."""

from pathlib import Path
import numpy as np
from room_for_doubt.evaluation.selection import paired_accuracy_interval


def target_contrasts(output_dir, seeds):
    out = Path(output_dir)
    rows = []
    for seed in seeds:
        bundles = {}
        for mode in ["last", "mean", "full"]:
            with np.load(out / f"artifacts/test_{mode}_{seed}.npz") as f:
                bundles[mode] = {
                    k: f[k] for k in ["labels", "corrected", "weighted_probabilities"]
                }
        labels = bundles["full"]["labels"]
        if any(not np.array_equal(labels, b["labels"]) for b in bundles.values()):
            raise ValueError("Pairwise labels differ")
        for left, right in [("last", "mean"), ("mean", "full"), ("last", "full")]:
            for action in ["top2", "uw_tta"]:
                pred = lambda mode: (
                    bundles[mode]["corrected"]
                    if action == "top2"
                    else bundles[mode]["weighted_probabilities"].argmax(1)
                )
                rows.append(
                    dict(
                        seed=seed,
                        comparison=f"{right}_minus_{left}",
                        action=action,
                        **paired_accuracy_interval(labels, pred(left), pred(right)),
                    )
                )
    return rows
