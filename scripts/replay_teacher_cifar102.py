"""Verify CIFAR-10.2 contrasts and export diagnostics from saved predictions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from room_for_doubt.evaluation.diagnostic_metrics import detection_metrics, probability_metrics
from room_for_doubt.evaluation.temporal_teacher import METHODS, SEEDS, contrast
from room_for_doubt.evaluation.tie_aware_risk import expected_tie_aurc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    args = parser.parse_args()
    root = args.evaluation
    expected = json.loads((root / "replay.json").read_text())
    methods = {method: [] for method in METHODS}
    rows = []
    labels, source_ids = None, None
    for seed in SEEDS:
        path = root / f"seed{seed}_predictions.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["local_archive_sha256"][path.name]:
            raise ValueError(f"Archive hash mismatch for seed {seed}")
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["ids"], np.arange(expected["examples"]))
            if labels is None:
                labels, source_ids = saved["labels"], saved["source_ids"]
                np.testing.assert_array_equal(np.bincount(labels), np.full(10, 200))
                if len(np.unique(source_ids)) != len(labels):
                    raise ValueError("Repeated source IDs")
            np.testing.assert_array_equal(saved["labels"], labels)
            np.testing.assert_array_equal(saved["source_ids"], source_ids)
            base_correct = saved["base"].argmax(1) == labels
            for method in METHODS:
                p = saved[method]
                if p.shape != (len(labels), 10):
                    raise ValueError(f"Unexpected prediction shape for {seed}/{method}")
                guess = p.argmax(1)
                correct = guess == labels
                confidence = p.max(1)
                rows.append(dict(
                    seed=seed, method=method, **probability_metrics(labels, p),
                    aurc=expected_tie_aurc(confidence, ~correct),
                    own_error_auroc=detection_metrics(~correct, 1 - confidence)["auroc"],
                    fixed_base_error_auroc=detection_metrics(~base_correct, 1 - confidence)["auroc"],
                    corrected=int((correct & ~base_correct).sum()),
                    harmed=int((~correct & base_correct).sum()),
                ))
                methods[method].append(correct)

    pairs = {
        "early5_vs_static5": ("direct_soft5", "static5_direct"),
        "early5_vs_base": ("direct_soft5", "base"),
        "early5_vs_selected1": ("direct_soft5", "selected1_direct"),
        "static5_vs_selected1": ("static5_direct", "selected1_direct"),
        "early_teacher_vs_static_teacher": ("early5_teacher", "static5_teacher"),
    }
    for name, pair in pairs.items():
        result = contrast(methods, *pair, labels)
        for key, value in result.items():
            np.testing.assert_allclose(value, expected["contrasts"][name][key], rtol=0, atol=1e-12)
    with (root / "replayed_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} prediction rows and {len(pairs)} paired contrasts verified")


if __name__ == "__main__":
    main()
