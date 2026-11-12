"""Generate the external-test diagnostic tables from replayed predictions."""

import csv
import json
from pathlib import Path
from statistics import mean

from build_teacher_cifar102 import METHODS as TEACHER_METHODS, SEEDS

PAPER = Path(__file__).resolve().parent
S4_METHODS = {
    "base": "Base",
    "early5": "Early-5 residual",
    "direct_soft5": "Direct distillation",
    "early5_teacher": "Five-snapshot teacher",
}
METRICS = ("ece", "nll", "brier", "own_error_auroc", "fixed_base_error_auroc", "aurc")


def read_rows(path, methods):
    with path.open() as handle:
        records = list(csv.DictReader(handle))
    rows = {(int(row["seed"]), row["method"]): row for row in records}
    expected = {(seed, method) for seed in SEEDS for method in methods}
    if len(records) != len(expected) or set(rows) != expected:
        raise ValueError(f"Unexpected seed/method grid: {path}")
    return rows


def write_table(name, columns, header, body):
    lines = [rf"\begin{{tabular}}{{{columns}}}", r"\toprule", header,
             r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
    (PAPER / "revision/tables" / f"{name}.tex").write_text("\n".join(lines) + "\n")


def main():
    s4 = PAPER / "evidence/confirmation_s4"
    rows = read_rows(s4 / "metrics.csv", S4_METHODS)
    latency = json.loads((s4 / "observed_summary.json").read_text())["pipeline_batch1_ms"]
    body = []
    for method, label in S4_METHODS.items():
        scores = [100 * float(rows[seed, method]["acc"]) for seed in SEEDS]
        values = [*scores, mean(scores), latency[method]]
        body.append(" & ".join([label, *(f"{v:.2f}" for v in values)]) + r" \\")
    write_table("confirmation_s4", "lrrrrr", r"Method & 42 & 43 & 44 & Mean & ms \\", body)

    teacher = read_rows(PAPER / "evidence/teacher_cifar102/diagnostics.csv", TEACHER_METHODS)
    body = []
    for dataset, methods, records in (
        ("CIFAR-10.1 v6", S4_METHODS, rows),
        ("CIFAR-10.2", TEACHER_METHODS, teacher),
    ):
        body.append(rf"\multicolumn{{9}}{{l}}{{\textbf{{{dataset}}}}} \\")
        for method, label in methods.items():
            group = [records[seed, method] for seed in SEEDS]
            values = [f"{mean(float(row[key]) for row in group):.3f}" for key in METRICS]
            counts = [str(sum(int(row[key]) for row in group)) for key in ("corrected", "harmed")]
            body.append(" & ".join([label, *values, *counts]) + r" \\")
        body.append(r"\addlinespace")
    write_table("external_diagnostics", "lrrrrrrrr",
                r"Predictor & ECE & NLL & Brier & Own AUROC & Base AUROC & AURC & Corr. & Harm \\",
                body)


if __name__ == "__main__":
    main()
