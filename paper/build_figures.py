"""Render manuscript figures from compact, preserved experiment evidence.

No model fitting, policy selection, or test prediction is performed here.
"""

import csv
import hashlib
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "revision" / "figures"
RUNS = [("c100_s42", "CIFAR-100N / base 42"),
        ("c100_s43", "CIFAR-100N / base 43"),
        ("c10_s42", "CIFAR-10N Worse / base 42")]
METHODS = [("tta", "Uniform TTA", "#777777"),
           ("base_resnet50__best_margin", "Static target", "#D89022"),
           ("base_resnet50__history_mean", "History-50", "#8D68A6"),
           ("base_resnet50__early_mean", "Early-5", "#007F83")]


def read(run, name):
    with (ROOT / "evidence" / run / (name + ".csv")).open() as stream:
        return list(csv.DictReader(stream))


def check_evidence():
    with (ROOT / "evidence/file_manifest.csv").open() as stream:
        for row in csv.DictReader(stream):
            path = ROOT / row["file"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path


def members(run, method):
    rows = [r for r in read(run, "test_validation_selected")
            if r["method"] == method or r["method"].startswith(method + "__")]
    assert len(rows) == (3 if "__" in method else 1)
    aggregate = next(r for r in read(run, "aggregate") if r["method"] == method)
    assert abs(statistics.mean(float(r["acc"]) for r in rows)
               - float(aggregate["acc"])) < 1e-10
    return rows


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(DEST / (name + "." + ext), dpi=220, bbox_inches="tight",
                    metadata={"Creator": "Room for Doubt / paper/build_figures.py",
                              "CreationDate": None, "ModDate": None}
                    if ext == "pdf" else None)
    plt.close(fig)


def style(ax, axis="x"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color="#E6E8EB", linewidth=0.65)
    ax.set_axisbelow(True)


def accuracy_gain():
    fig, ax = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    for k, (method, label, color) in enumerate(METHODS):
        for i, (run, _) in enumerate(RUNS):
            base = float(next(r for r in read(run, "aggregate")
                              if r["method"] == "base")["acc"])
            vals = np.array([100 * (float(r["acc"]) - base)
                             for r in members(run, method)])
            y = i + (k - 1.5) * 0.17
            ax.errorbar(vals.mean(), y, xerr=vals.std(ddof=1) if len(vals) > 1 else 0,
                        fmt="o", color=color, capsize=3, markersize=5,
                        label=label if i == 0 else None, zorder=3)
            if len(vals) > 1:
                ax.scatter(vals, y + np.array([-.035, 0, .035]), s=10,
                           color=color, alpha=.5, zorder=2)
    ax.axvline(0, color="#30343B", linewidth=.9)
    ax.set_yticks(range(3), [label for _, label in RUNS])
    ax.set_ylim(2.55, -.65)
    ax.set_xlim(-.6, 2.0)
    ax.set_xlabel("Test accuracy gain over the same base (percentage points)")
    ax.set_title("Validation-selected policies on three completed bases", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False,
               loc="outside upper center", fontsize=8)
    style(ax)
    save(fig, "accuracy_gain")


def history_windows():
    aggregate = {r["method"]: r for r in read("c100_s42", "aggregate")}
    targets = ["early_mean_2", "early_mean", "early_mean_10", "early_mean_20",
               "history_mean", "late_mean_5"]
    labels = ["First 2", "First 5", "First 10", "First 20", "All 50", "Last 5"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), layout="constrained")
    for encoder, label, color in [("resnet18", "Frozen ResNet18", "#597BA7"),
                                   ("base_resnet50", "Selected base features", "#007F83")]:
        for ax, key, scale in [(axes[0], "acc", 100), (axes[1], "error_auroc", 1)]:
            vals = [scale * float(aggregate[encoder + "__" + t][key]) for t in targets]
            ax.plot(range(6), vals, "o-", color=color, linewidth=1.5,
                    markersize=4, label=label)
            ax.axhline(scale * float(aggregate[encoder + "__best_margin"][key]),
                       color=color, linestyle="--", linewidth=.9, alpha=.8)
    for ax in axes:
        ax.set_xticks(range(6), labels, rotation=35, ha="right")
        ax.set_xlabel("Summary target (categorical)")
        style(ax, "y")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].set_ylabel("Original-view base-error AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="#555555", linestyle="--", label="Static target"))
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=8)
    save(fig, "history_windows")


def calibration_changes():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True, layout="constrained")
    for ax, key, title in zip(axes, ["ece", "nll"], ["Change in ECE", "Change in NLL"]):
        for k, (method, label, color) in enumerate(METHODS):
            for i, (run, _) in enumerate(RUNS):
                base = next(r for r in read(run, "test_validation_selected") if r["method"] == "base")
                delta = statistics.mean(float(r[key]) for r in members(run, method)) - float(base[key])
                ax.scatter(delta, i + (k - 1.5) * .17, color=color, s=24,
                           label=label if i == 0 else None, zorder=3)
        ax.axvline(0, color="#30343B", linewidth=.9)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Difference from base (lower is better)")
        style(ax)
    axes[0].set_yticks(range(3), ["C100 / base 42", "C100 / base 43", "C10 Worse / base 42"])
    axes[0].set_ylim(2.5, -.55)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, frameon=False, fontsize=8)
    save(fig, "calibration_changes")


def main():
    check_evidence()
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    accuracy_gain()
    history_windows()
    calibration_changes()
    print("Generated 3 figures (PDF + PNG); evidence hashes and head means verified.")


if __name__ == "__main__":
    main()
