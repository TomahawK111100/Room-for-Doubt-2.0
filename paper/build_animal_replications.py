"""Render three independent Animal bases; never pool head seeds as base replicates."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from build_figures import check_evidence

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "evidence/animal10n_replications"
DEST = ROOT / "revision"


def main():
    check_evidence()
    per_base = pd.read_csv(SOURCE / "per_base_methods.csv")
    selected = pd.read_csv(SOURCE / "per_head_selected.csv")
    aggregate = pd.read_csv(SOURCE / "aggregate.csv")
    audit = json.loads((SOURCE / "audit_metrics.json").read_text())
    assert audit["independent_bases"] == 3 and audit["heads_per_base"] == 3
    assert set(per_base.base_seed) == {42, 43, 44}
    for _, row in aggregate.iterrows():
        data = per_base[per_base.method == row["method"]]
        assert len(data) == 3 and abs(data.acc_pct.mean() - row.acc_pct) < 1e-10
        assert abs(data.gain_vs_tta_pp.mean() - row.gain_vs_tta_pp) < 1e-10
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & Base42 & Base43 & Base44 & Mean & Error AUROC \\",
        r"\midrule",
    ]
    for _, row in aggregate.iterrows():
        data = per_base[per_base.method == row["method"]].set_index("base_seed")
        lines.append(
            f"{row['label']} & {data.loc[42, 'acc_pct']:.2f} & {data.loc[43, 'acc_pct']:.2f} & {data.loc[44, 'acc_pct']:.2f} & {row.acc_pct:.2f} & {row.auroc:.4f} "
            + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DEST / "tables/animal10n_replications.tex").write_text("\n".join(lines) + "\n")
    methods = [
        ("msp", "MSP"),
        ("base_resnet50__best_margin", "Static"),
        ("base_resnet50__best_augmented_mean", "Augmented static"),
        ("base_resnet50__history_mean", "History-50"),
        ("base_resnet50__early_mean", "Early-5"),
    ]
    colors = {42: "#087f8c", 43: "#b56620", 44: "#6f5bae"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, (method, label) in enumerate(methods):
        for seed in [42, 43, 44]:
            part = selected[
                (selected.base_seed == seed)
                & (
                    (selected.method == method)
                    | selected.method.str.startswith(method + "__")
                )
            ]
            base = selected[
                (selected.base_seed == seed) & (selected.method == "base")
            ].iloc[0]
            values = [
                part.gain_pp_vs_tta_selected.to_numpy(),
                part.base_error_auroc.to_numpy() - base.base_error_auroc,
            ]
            y = i + (seed - 43) * 0.19
            for ax, points in zip(axes, values):
                ax.scatter(
                    points, np.full(len(points), y), s=18, c=colors[seed], alpha=0.5
                )
                ax.scatter(
                    points.mean(),
                    y,
                    s=45,
                    marker="D",
                    facecolors="white",
                    edgecolors=colors[seed],
                    zorder=4,
                )
    for ax in axes:
        ax.set_yticks(range(len(methods)), [m[1] for m in methods])
        ax.invert_yaxis()
        ax.axvline(0, color="#777", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_xlabel("Accuracy gain vs selected uniform TTA (pp)")
    axes[1].set_xlabel("Original-view error AUROC difference vs MSP")
    fig.legend(
        [
            Line2D(
                [],
                [],
                marker="D",
                linestyle="",
                color=colors[s],
                markerfacecolor="white",
            )
            for s in [42, 43, 44]
        ],
        ["Base42", "Base43", "Base44"],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.91),
        frameon=False,
    )
    fig.suptitle(
        "Animal-10N: three independent bases, three nested head seeds per base",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.01,
        "Dots: nested heads; diamonds: within-base means. No between-base confidence interval is implied.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    for ext in ["png", "pdf"]:
        fig.savefig(
            DEST / "figures" / ("animal10n_replications." + ext),
            dpi=180,
            bbox_inches="tight",
        )
    plt.close(fig)
    print("Three-base Animal table and figures rebuilt from hash-verified evidence.")


if __name__ == "__main__":
    main()
