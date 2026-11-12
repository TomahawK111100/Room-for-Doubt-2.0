"""Generate the CIFAR-10.2 manuscript table from the compact replay evidence."""

import csv
import json
from math import isclose
from pathlib import Path
from statistics import mean

PAPER = Path(__file__).resolve().parent
EVIDENCE = PAPER / "evidence/teacher_cifar102"
SEEDS = (42, 43, 44)
METHODS = {
    "base": "Base",
    "direct_soft5": "Early-5 direct",
    "static5_direct": "Static-5 direct",
    "selected1_direct": "Selected-single direct",
    "early5_teacher": "Early-5 teacher",
    "static5_teacher": "Static-5 teacher",
}
CONTRASTS = {
    "early5_vs_static5": ("direct_soft5", "static5_direct"),
    "early5_vs_base": ("direct_soft5", "base"),
    "early5_vs_selected1": ("direct_soft5", "selected1_direct"),
    "static5_vs_selected1": ("static5_direct", "selected1_direct"),
    "early_teacher_vs_static_teacher": ("early5_teacher", "static5_teacher"),
}


def main():
    summary = json.loads((EVIDENCE / "replay.json").read_text())
    with (EVIDENCE / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    accuracy = {(int(row["seed"]), row["method"]): 100 * float(row["acc"]) for row in rows}
    expected = {(seed, method) for seed in SEEDS for method in METHODS}
    if len(rows) != len(expected) or set(accuracy) != expected:
        raise ValueError("Expected six predictors for each of the three fitted bases")
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Predictor & 42 & 43 & 44 & Mean \\",
        r"\midrule",
    ]
    for method, name in METHODS.items():
        scores = [accuracy[seed, method] for seed in SEEDS]
        average = mean(scores)
        if not isclose(average, summary["mean_accuracy_percent"][method], abs_tol=1e-10):
            raise ValueError(f"Replay mean differs for {method}")
        lines.append(" & ".join([name, *(f"{score:.2f}" for score in scores), f"{average:.2f}"]) + r" \\")
    for name, (method, baseline) in CONTRASTS.items():
        replay = summary["contrasts"][name]
        deltas = [accuracy[seed, method] - accuracy[seed, baseline] for seed in SEEDS]
        if any(not isclose(a, b, abs_tol=1e-10) for a, b in zip(deltas, replay["per_seed_pp"], strict=True)):
            raise ValueError(f"Replay deltas differ for {name}")
        if not isclose(mean(deltas), replay["mean_pp"], abs_tol=1e-10):
            raise ValueError(f"Replay mean differs for {name}")
    table = PAPER / "revision/tables/teacher_cifar102.tex"
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    table.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
