"""Render follow-up tables and a paired-control figure from preserved compact CSVs."""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_figures import ROOT, check_evidence, save, style
from build_tables import table

RUNS = [("c100_s44", "C100/44"), ("c10_s43", "C10/43")]
LABELS = {
    "best_margin": "Static",
    "best_augmented_mean": "Aug-static",
    "history_mean": "History-50",
    "early_mean": "Early-5",
}


def read(run, stage, name):
    path = ROOT / "evidence/followups" / f"{run}_{stage}_{name}.csv"
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    check_evidence()
    rows = []
    for mode in ("best_margin", "best_augmented_mean", "history_mean", "early_mean"):
        cells = [LABELS[mode]]
        for run, _ in RUNS:
            r = next(r for r in read(run, "augmentation", "aggregate")
                     if r["method"] == "base_resnet50__" + mode)
            cells.append(f"{100*float(r['acc']):.2f} $\\pm$ {100*float(r['acc_sd']):.2f}")
        rows.append(cells)
    table("followup_accuracy.tex", "lrr", "Target & C100 (44) & C10 Worse (43)", rows)

    rows = []
    for run, label in RUNS:
        for r in read(run, "augmentation", "paired_target_contrasts"):
            assert r["encoder"] == "base_resnet50"
            rows.append([
                label, r["head_seed"], LABELS[r["reference"]],
                r["reference_views"] + "/" + r["proposed_views"],
                f"{float(r['gain_pp']):+.2f}",
                f"[{float(r['ci_low_pp']):+.2f}, {float(r['ci_high_pp']):+.2f}]",
                f"{float(r['auroc_difference']):+.4f}",
            ])
    table("followup_pairs.tex", "lllrrrr",
          "Run & Seed & Reference & $V_r/V_e$ & Gain & 95\\% CI & $\\Delta$AUROC", rows)

    rows = []
    for run, label in RUNS:
        source = read(run, "runtime", "runtime_cost")
        for method, views in (("base", 1), ("uniform", 5), ("early", 5),
                              ("uniform", 10), ("early", 10)):
            selected = [r for r in source if r["method"] == method
                        and int(r["views"]) == views and int(r["batch_images"]) == 1]
            assert len(selected) == 2
            scopes = {r["scope"]: r for r in selected}
            gpu, total = scopes["model_only"], scopes["with_preprocessing"]
            rows.append([
                label, dict(base="Base", uniform="Uniform", early="Early-5")[method],
                str(views), f"{float(gpu['batch_ms_median']):.2f}",
                f"{float(total['batch_ms_median']):.2f}",
                f"{float(total['peak_cuda_allocated_mb']):.1f}",
            ])
    table("followup_runtime.tex", "llrrrr",
          "Run & Method & Views & Model ms & With prep. ms & Peak MB", rows)

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    index = 0
    labels = []
    for run, title in RUNS:
        for r in read(run, "augmentation", "paired_target_contrasts"):
            if r["reference"] != "best_augmented_mean":
                continue
            gain, low, high = [float(r[k]) for k in ("gain_pp", "ci_low_pp", "ci_high_pp")]
            color = "#007F83" if run.startswith("c100") else "#597BA7"
            ax.errorbar(gain, index, xerr=[[gain-low], [high-gain]],
                        fmt="o", color=color, capsize=4, markersize=5)
            labels.append(title + " / head " + r["head_seed"])
            index += 1
    ax.set_yticks(range(index), labels)
    ax.set_ylim(index-.5, -.5)
    ax.axvline(0, color="#444444", linewidth=1)
    ax.set_xlabel("Early-5 minus augmented static accuracy (percentage points)")
    ax.set_title("Conditional paired 95% intervals; each method keeps its own policy",
                 loc="left", fontsize=10)
    style(ax)
    save(fig, "augmentation_control")
    print("Generated 3 follow-up tables and paired-control figure; evidence hashes verified.")


if __name__ == "__main__":
    main()
