"""Render follow-up tables and a paired-control figure from preserved compact CSVs."""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_figures import ROOT, check_evidence, save, style
from build_tables import table

RUNS = [("c100_s42", "C100/42"), ("c100_s43", "C100/43"),
        ("c100_s44", "C100/44"), ("c10_s42", "C10/42"),
        ("c10_s43", "C10/43"), ("c10_s44", "C10/44")]
RUNTIME_RUNS = [("c100_s44", "C100/44"), ("c10_s43", "C10/43")]
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
    for prefix, dataset in (("c100_", "CIFAR-100N"), ("c10_", "CIFAR-10N Worse")):
        group = [(run, label) for run, label in RUNS if run.startswith(prefix)]
        rows.append([r"\multicolumn{4}{l}{\textbf{" + dataset + "}}"])
        for mode in ("best_margin", "best_augmented_mean", "history_mean", "early_mean"):
            cells = [LABELS[mode]]
            for run, _ in group:
                r = next(r for r in read(run, "augmentation", "aggregate")
                         if r["method"] == "base_resnet50__" + mode)
                cells.append(f"{100*float(r['acc']):.2f} $\\pm$ {100*float(r['acc_sd']):.2f}")
            rows.append(cells)
    table("followup_accuracy.tex", "lrrr", "Target & Base 42 & Base 43 & Base 44", rows)

    for prefix, filename in (("c100_", "followup_pairs.tex"),
                             ("c10_", "followup_pairs_c10.tex")):
        rows = []
        for run, label in RUNS:
            if not run.startswith(prefix):
                continue
            for r in read(run, "augmentation", "paired_target_contrasts"):
                assert r["encoder"] == "base_resnet50"
                rows.append([
                    label, r["head_seed"], LABELS[r["reference"]],
                    r["reference_views"] + "/" + r["proposed_views"],
                    f"{float(r['gain_pp']):+.2f}",
                    f"[{float(r['ci_low_pp']):+.2f}, {float(r['ci_high_pp']):+.2f}]",
                    f"{float(r['auroc_difference']):+.4f}",
                ])
        table(filename, "lllrrrr",
              "Run & Seed & Reference & $V_r/V_e$ & Gain & 95\\% CI & $\\Delta$AUROC", rows)

    rows = []
    for run, label in RUNTIME_RUNS:
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
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.2), sharex=True,
                             layout="constrained")
    for ax, prefix, title, color in zip(
            axes, ("c100_", "c10_"), ("CIFAR-100N", "CIFAR-10N Worse"),
            ("#007F83", "#597BA7")):
        labels = []
        for run, _ in RUNS:
            if not run.startswith(prefix):
                continue
            for r in read(run, "augmentation", "paired_target_contrasts"):
                if r["reference"] != "best_augmented_mean":
                    continue
                gain, low, high = [float(r[k]) for k in ("gain_pp", "ci_low_pp", "ci_high_pp")]
                index = len(labels)
                ax.errorbar(gain, index, xerr=[[gain-low], [high-gain]],
                            fmt="o", color=color, capsize=4, markersize=5)
                labels.append(run.split("s")[-1] + " / " + r["head_seed"])
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels)-.5, -.5)
        ax.set_xlim(-.3, 2.05)
        ax.axvline(0, color="#444444", linewidth=1)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("Base seed / head seed")
        style(ax)
    fig.supxlabel("Early-5 minus augmented static accuracy (percentage points)", fontsize=9)
    fig.suptitle("Conditional paired 95% intervals; each method keeps its own policy", fontsize=10)
    save(fig, "augmentation_control")
    print("Generated 4 follow-up tables and paired-control figure; evidence hashes verified.")


if __name__ == "__main__":
    main()
