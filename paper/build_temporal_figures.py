"""Rebuild the four historical posterior-compression figures from compact evidence."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parent
EVIDENCE = PAPER / "evidence"
DEST = PAPER / "revision/figures"


def read(study, name):
    with (EVIDENCE / study / f"{name}.csv").open() as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    for suffix in ("pdf", "png"):
        fig.savefig(DEST / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def bars(ax, labels, values):
    ax.bar(range(len(values)), values, color="#347a99")
    ax.set_xticks(range(len(values)), labels, rotation=14, ha="right")
    ax.set_ylabel("Accuracy (%)")
    for x, value in enumerate(values):
        ax.text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(values) * 1.16)


def intervals(ax, rows, labels):
    for y, row in enumerate(rows):
        point, low, high = (float(row[key]) for key in ("gain_pp", "ci_low_pp", "ci_high_pp"))
        ax.plot([low, high], [y, y], color="#347a99", lw=1.6)
        ax.scatter([point], [y], color="#194d64", zorder=2)
    ax.axvline(0, color="0.55", ls="--", lw=1)
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy difference (percentage points)")


def main():
    name = "temporal_residual_validation"
    methods = read(name, "methods")
    history = read(name, "head_history")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    bars(axes[0], ["Base", "Residual", "Teacher"], [float(row["val_noisy_acc_pct"]) for row in methods])
    for key, label in (("train_kl", "Fit"), ("meta_val_kl", "Meta-val")):
        axes[1].plot([int(row["epoch"]) for row in history], [float(row[key]) for row in history], label=label)
    axes[1].set(xlabel="Head epoch", ylabel="KL divergence")
    for file, label in (("validation_policies", "Residual"), ("validation_teacher_interpolation", "Teacher")):
        grid = read(name, file)
        axes[2].plot([float(row["alpha"]) for row in grid],
                     [100 * float(row["val_noisy_acc"]) for row in grid], "o-", label=label)
    axes[2].set(xlabel="Mixing strength", ylabel="Noisy-val accuracy (%)")
    for ax in axes[1:]:
        ax.legend(frameon=False, fontsize=8)
    save(fig, name)

    name = "temporal_residual_test"
    methods = read(name, "methods")[:3]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
    bars(axes[0], ["Base", "Residual", "Teacher"], [100 * float(row["acc"]) for row in methods])
    intervals(axes[1], read(name, "paired_comparisons"),
              ["Residual − base", "Teacher − base", "Residual − teacher"])
    save(fig, name)

    groups = (
        ("temporal_residual_replications", {
            "temporal_minus_base": "Early-5 − base",
            "single_epoch1_minus_base": "Epoch 1 − base",
            "temporal_minus_single_epoch1": "Early-5 − epoch 1",
        }),
        ("temporal_selected_control", {
            "selected_student_minus_base": "Selected student − base",
            "temporal_minus_selected_student": "Early-5 − selected student",
            "temporal_minus_selected_teacher_interpolation": "Early-5 − two-pass interpolation",
        }),
    )
    for name, names in groups:
        rows = sorted(read(name, "paired_comparisons"), key=lambda row: int(row["seed"]))
        fig, ax = plt.subplots(figsize=(8.7, 3), layout="constrained")
        intervals(ax, rows, [f"{row['seed']}: {names[row['comparison']]}" for row in rows])
        save(fig, name)


if __name__ == "__main__":
    main()
