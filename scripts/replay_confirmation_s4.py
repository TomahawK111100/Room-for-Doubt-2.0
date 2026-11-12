"""Replay S4 metrics from saved predictions without running a model."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from room_for_doubt.data.confirmation import METHODS, SEEDS
from room_for_doubt.evaluation.confirmation import paired_stratified_interval
from room_for_doubt.evaluation.diagnostic_metrics import detection_metrics, probability_metrics
from room_for_doubt.evaluation.tie_aware_risk import expected_tie_aurc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    root = args.evaluation
    summary = json.loads((root / "observed_summary.json").read_text())
    with (root / "accuracy.csv").open() as handle:
        reference = {int(row["seed"]): row for row in csv.DictReader(handle)}

    labels = None
    rows = []
    correctness = {method: [] for method in METHODS}
    for seed in SEEDS:
        with np.load(root / f"seed{seed}_predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["ids"], np.arange(2000)):
                raise ValueError(f"Unexpected image IDs for seed {seed}")
            if labels is None:
                labels = saved["labels"]
                np.testing.assert_array_equal(np.bincount(labels), np.full(10, 200))
            np.testing.assert_array_equal(saved["labels"], labels)
            base_correct = saved["base"].argmax(1) == labels
            for method in METHODS:
                probs = saved[method]
                if probs.shape != (2000, 10):
                    raise ValueError(f"Unexpected prediction shape: {seed}, {method}")
                correct = probs.argmax(1) == labels
                metrics = probability_metrics(labels, probs)
                np.testing.assert_allclose(metrics["acc"], float(reference[seed][method]),
                                           rtol=0, atol=1e-12)
                correctness[method].append(correct.astype(float))
                confidence = probs.max(1)
                rows.append(dict(
                    seed=seed, method=method, **metrics,
                    aurc=expected_tie_aurc(confidence, ~correct),
                    own_error_auroc=detection_metrics(~correct, 1 - confidence)["auroc"],
                    fixed_base_error_auroc=detection_metrics(~base_correct, 1 - confidence)["auroc"],
                    corrected=int((correct & ~base_correct).sum()),
                    harmed=int((~correct & base_correct).sum()),
                ))

    correct = {method: np.stack(values) for method, values in correctness.items()}
    contrasts = {}
    for name, method, reference_method in (
        ("early5_vs_base", "early5", "base"),
        ("teacher_vs_early5", "early5_teacher", "early5"),
        ("direct_vs_base", "direct_soft5", "base"),
        ("direct_vs_early5", "direct_soft5", "early5"),
    ):
        differences = correct[method] - correct[reference_method]
        gains = differences.mean(axis=1) * 100
        contrasts[name] = dict(per_seed_pp=gains.tolist(), mean_pp=float(gains.mean()),
                               between_seed_sd_pp=float(gains.std(ddof=1)),
                               paired_ci95_pp=paired_stratified_interval(differences, labels))

    primary = contrasts["early5_vs_base"]
    teacher = contrasts["teacher_vs_early5"]
    for actual, expected in (
        (primary["per_seed_pp"], summary["base_gain_pp"]),
        (primary["mean_pp"], summary["mean_gain_pp"]),
        (primary["between_seed_sd_pp"], summary["gain_between_base_sd_pp"]),
        (primary["paired_ci95_pp"], summary["gain_ci95_pp"]),
        (teacher["mean_pp"], summary["teacher_gap_pp"]),
        (teacher["paired_ci95_pp"], summary["teacher_gap_ci95_pp"]),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    with (root / "replayed_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (root / "replayed_contrasts.json").write_text(json.dumps(contrasts, indent=2) + "\n")
    print(json.dumps(contrasts, indent=2))


if __name__ == "__main__":
    main()
