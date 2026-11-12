"""Summarize the completed Mini-WebVision pilot without fitting or selection."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "evidence/miniwebvision_s42"
METHODS = [
    ("base", "FP32 base"),
    ("tta", "Uniform TTA"),
    ("base_resnet50__best_margin", "Static"),
    ("base_resnet50__best_augmented_mean", "Augmented static"),
    ("base_resnet50__history_mean", "History-50"),
    ("base_resnet50__early_mean", "Early-5"),
]


def read(name):
    with (SOURCE / name).open() as stream:
        return list(csv.DictReader(stream))


def members(rows, prefix):
    return [r for r in rows if r["method"] == prefix or r["method"].startswith(prefix + "__")]


def main():
    rows = read("selected.csv")
    study = {r["method"]: r for r in read("study_selected.csv")}
    assert len({r["method"] for r in rows}) == len(rows)
    for row in rows:
        if row["method"] in study:
            assert row == study[row["method"]], "Shared results changed in control stage"
    history = read("history.csv")
    assert [int(r["epoch"]) for r in history] == list(range(50))
    run = json.loads((SOURCE / "run_metrics.json").read_text())
    assert run["status"] == "complete" and all(s["status"] == "complete" for s in run["stages"])
    base = json.loads((SOURCE / "base_metrics.json").read_text())
    assert base["completed_epochs"] == 50
    summary = []
    for prefix, label in METHODS:
        part = members(rows, prefix)
        assert len(part) == (1 if prefix in ("base", "tta") else 3)
        assert all(r["selected_by_validation"] == "True" for r in part)
        row = {"method": prefix, "label": label, "heads": len(part)}
        for key in ("acc", "base_error_auroc", "ece", "nll"):
            row[key] = statistics.mean(float(r[key]) for r in part)
        row["acc_pct"] = 100 * row["acc"]
        summary.append(row)
    with (SOURCE / "selected_methods.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    early = members(rows, "base_resnet50__early_mean")
    paired = read("paired.csv")
    metrics = {
        "independent_bases": 1,
        "base_seed": 42,
        "nested_head_seeds": [42, 43, 44],
        "selected_results": summary,
        "early_gain_over_tta_pp": 100 * (summary[-1]["acc"] - summary[1]["acc"]),
        "early_gain_over_augmented_static_pp": 100 * (summary[-1]["acc"] - summary[3]["acc"]),
        "early_gain_over_history_pp": 100 * (summary[-1]["acc"] - summary[4]["acc"]),
        "positive_conditional_accuracy_intervals_vs_tta": sum(float(r["ci_low_pp_vs_tta_selected"]) > 0 for r in early),
        "positive_conditional_accuracy_intervals_vs_augmented_static": sum(float(r["ci_low_pp"]) > 0 for r in paired if r["reference"] == "best_augmented_mean"),
        "positive_conditional_accuracy_intervals_vs_history": sum(float(r["ci_low_pp"]) > 0 for r in paired if r["reference"] == "history_mean"),
        "resumed_stage_hours_excluding_data_preparation": run["elapsed_seconds"] / 3600,
        "amp_training_test_acc": base["test_acc"],
        "fp32_downstream_base_acc": summary[0]["acc"],
        "raw_prediction_replay_performed": False,
    }
    (SOURCE / "analysis_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    dest = ROOT / "revision"
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Method & Acc. (\%) & Error AUROC & ECE \\", r"\midrule"]
    for row in summary:
        lines.append(f"{row['label']} & {row['acc_pct']:.2f} & {row['base_error_auroc']:.4f} & {row['ece']:.4f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (dest / "tables/miniwebvision_pilot.tex").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for axis, key, factor, xlabel in [(axes[0], "acc", 100, "Final evaluation accuracy (%)"), (axes[1], "base_error_auroc", 1, "Original-view base-error AUROC")]:
        for i, (prefix, label) in enumerate(METHODS):
            values = [factor * float(r[key]) for r in members(rows, prefix)]
            color = "#087f8c" if label == "Early-5" else "#687786"
            axis.scatter(values, [i] * len(values), color=color, s=24, alpha=0.65)
            axis.scatter(statistics.mean(values), i, marker="D", s=55, facecolors="white", edgecolors=color, zorder=3)
        axis.set_yticks(range(len(METHODS)), [label for _, label in METHODS])
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Mini-WebVision: one base (seed42), three nested head seeds")
    fig.text(0.5, 0.015, "Dots: heads; diamonds: means. No between-base uncertainty is estimated.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(dest / f"figures/miniwebvision_pilot.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
