"""Rebuild S2 comparison table/figure from compact evidence, without weights."""

import csv
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence/posthoc_controls"
LABELS = {
    "base": "Base",
    "selected1": "Selected single",
    "supervised_head": "Noisy-label head",
    "early2": "Early-2 residual",
    "early5": "Early-5 residual",
    "direct_soft5": "Early-5 direct",
    "independent2": "Independent-2 student",
}
ORDER = list(LABELS)


def build():
    with (EVIDENCE / "methods.csv").open() as f:
        rows = list(csv.DictReader(f))
    values = {(int(r["seed"]), r["method"]): float(r["acc"]) * 100 for r in rows}
    with (EVIDENCE / "paired_comparisons.csv").open() as f:
        pairs = list(csv.DictReader(f))
    with (EVIDENCE / "aggregate.csv").open() as f:
        agg = {r["method"]: r for r in csv.DictReader(f)}
    table = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Exploratory CIFAR-100N controls: accuracy (\\%). All student methods use one backbone pass. Means give equal weight to the three original bases; the head seed is fixed at42.}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method &42&43&44&Mean\\\\",
        "\\midrule",
    ]
    for method in ORDER:
        table.append(
            LABELS[method]
            + " & "
            + " & ".join(f"{values[s,method]:.2f}" for s in (42, 43, 44))
            + f" & {float(agg[method]['acc_mean'])*100:.2f}"
            + "\\\\"
        )
    table += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (ROOT / "revision/tables/posthoc_controls.tex").write_text("\n".join(table) + "\n")
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    colors = ["#3274a1", "#d08b29", "#3b8a64"]
    for seed, offset, color in zip((42, 43, 44), (-0.16, 0, 0.16), colors):
        a.scatter(
            np.arange(len(ORDER)) + offset,
            [values[seed, m] for m in ORDER],
            s=37,
            color=color,
            label=f"Base seed {seed}",
            zorder=3,
        )
    means = [float(agg[m]["acc_mean"]) * 100 for m in ORDER]
    a.scatter(
        np.arange(len(ORDER)),
        means,
        marker="_",
        s=250,
        color="#222222",
        zorder=4,
        label="Three-base mean",
    )
    a.set_xticks(range(len(ORDER)), [LABELS[m] for m in ORDER], rotation=32, ha="right")
    a.set_ylim(68.6, 72.45)
    a.set_ylabel("Test accuracy (%)")
    a.grid(axis="y", alpha=0.22)
    a.set_title("(a) Frozen student comparisons")
    a.legend(frameon=False, fontsize=8, loc="lower right")
    comparisons = [
        ("early5__minus__selected1", "Early-5 − selected single"),
        ("early5__minus__direct_soft5", "Early-5 − direct head"),
        ("early2__minus__independent2", "Early-2 − independent-2"),
    ]
    for i, (key, label) in enumerate(comparisons):
        for seed, offset, color in zip((42, 43, 44), (-0.19, 0, 0.19), colors):
            row = next(
                r for r in pairs if int(r["seed"]) == seed and r["comparison"] == key
            )
            x, lo, hi = (float(row[k]) for k in ["gain_pp", "ci_low_pp", "ci_high_pp"])
            b.errorbar(
                x,
                i + offset,
                xerr=[[x - lo], [hi - x]],
                fmt="o",
                color=color,
                capsize=2.5,
                markersize=4.5,
            )
    b.axvline(0, color="#555555", linewidth=0.9, linestyle="--")
    b.set_yticks(range(3), [x[1] for x in comparisons])
    b.invert_yaxis()
    b.set_xlim(-1.15, 1.05)
    b.grid(axis="x", alpha=0.2)
    b.set_xlabel("Accuracy difference (percentage points)")
    b.set_title("(b) Nominal paired-example 95% intervals")
    fig.text(
        0.5,
        0.012,
        "Known test reused; intervals condition on fitted models. Shared independent-teacher donors are correlated.",
        ha="center",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 1], w_pad=2)
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"revision/figures/posthoc_controls.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    build()
