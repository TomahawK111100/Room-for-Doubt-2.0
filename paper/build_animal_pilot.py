"""Render the single-base prospective Animal pilot from saved compact evidence."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from build_figures import check_evidence

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "evidence/animal10n_s42"
DEST = ROOT / "revision"


def main():
    check_evidence()
    summary = pd.read_csv(SOURCE / "selected_methods.csv")
    selected = pd.read_csv(SOURCE / "augmentation_test_validation_selected.csv")
    audit = json.loads((SOURCE / "audit_metrics.json").read_text())
    assert audit["independent_bases"] == 1 and audit["head_seeds"] == 3
    for _, r in summary.iterrows():
        part = selected[
            (selected.method == r["method"])
            | selected.method.str.startswith(r["method"] + "__")
        ]
        assert abs(100 * part.acc.mean() - r["acc_pct"]) < 1e-10
    (DEST / "tables").mkdir(exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Acc. (\%) & Error AUROC & ECE \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{r['label']} & {r.acc_pct:.2f} & {r.auroc:.4f} & {r.ece:.4f} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DEST / "tables/animal10n_pilot.tex").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, key, label in [
        (axes[0], "acc", "Test accuracy (%)"),
        (axes[1], "base_error_auroc", "Original-view base-error AUROC"),
    ]:
        for i, (_, r) in enumerate(summary.iterrows()):
            p = selected[
                (selected.method == r["method"])
                | selected.method.str.startswith(r["method"] + "__")
            ]
            factor = 100 if key == "acc" else 1
            values = p[key].to_numpy() * factor
            color = "#087f8c" if r["label"] == "Early-5" else "#687786"
            axis.scatter(values, [i] * len(values), s=25, c=color, alpha=0.65, zorder=3)
            axis.scatter(
                values.mean(),
                i,
                s=65,
                marker="D",
                facecolors="white",
                edgecolors=color,
                zorder=4,
            )
        axis.set_yticks(range(len(summary)), summary.label)
        axis.invert_yaxis()
        axis.set_xlabel(label)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_xlim(89, 90.4)
    axes[1].set_xlim(0.80, 0.93)
    fig.suptitle("Animal-10N: one base (seed42), three nested head seeds", fontsize=12)
    fig.text(
        0.5,
        0.01,
        "Dots: individual heads; diamonds: within-base means. Policies selected on noisy validation.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    for ext in ["png", "pdf"]:
        fig.savefig(
            DEST / "figures" / ("animal10n_pilot." + ext), dpi=180, bbox_inches="tight"
        )
    plt.close(fig)
    print("Animal pilot table and figures rebuilt from verified evidence.")


if __name__ == "__main__":
    main()
