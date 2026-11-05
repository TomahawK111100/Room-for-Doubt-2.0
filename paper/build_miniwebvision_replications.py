"""Summarize three completed bases, keeping nested heads within each base."""

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "evidence/miniwebvision_replications"
DEST = ROOT / "revision"
METHODS = [
    ("base", "FP32 base"),
    ("tta", "Uniform TTA"),
    ("base_resnet50__best_margin", "Static"),
    ("base_resnet50__best_augmented_mean", "Augmented static"),
    ("base_resnet50__history_mean", "History-50"),
    ("base_resnet50__early_mean", "Early-5"),
]


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def members(rows, prefix):
    return [r for r in rows if r["method"] == prefix or r["method"].startswith(prefix + "__")]


def intervals(rows, low, high):
    return {
        "positive": sum(float(r[low]) > 0 for r in rows),
        "negative": sum(float(r[high]) < 0 for r in rows),
        "touches_zero": sum(float(r[low]) == 0 or float(r[high]) == 0 for r in rows),
        "crosses_zero": sum(float(r[low]) < 0 < float(r[high]) for r in rows),
    }


def main():
    per_base, heads, contrasts = [], [], []
    runtime = {}
    for seed in (42, 43, 44):
        rows = read(SOURCE / f"seed{seed}_selected.csv")
        assert len({r["method"] for r in rows}) == len(rows)
        early = members(rows, "base_resnet50__early_mean")
        assert {r["method"].split("__")[-1] for r in early} == {"42", "43", "44"}
        tta = float(members(rows, "tta")[0]["acc"])
        base = float(members(rows, "base")[0]["acc"])
        heads.extend(dict(base_seed=seed, **r) for r in rows)
        contrasts.extend(dict(base_seed=seed, **r) for r in read(SOURCE / f"seed{seed}_paired.csv"))
        history = read(SOURCE / f"seed{seed}_history.csv")
        assert [int(r["epoch"]) for r in history] == list(range(50))
        run = json.loads((SOURCE / f"seed{seed}_run_metrics.json").read_text())
        assert run["status"] == "complete" and all(s["returncode"] == 0 for s in run["stages"])
        runtime[seed] = run["elapsed_seconds"] / 3600
        for prefix, label in METHODS:
            group = members(rows, prefix)
            assert len(group) == (1 if prefix in ("base", "tta") else 3)
            r = dict(base_seed=seed, method=prefix, label=label, head_count=len(group))
            r.update({k: statistics.mean(float(x[k]) for x in group) for k in ("acc", "base_error_auroc", "ece", "nll")})
            r.update(acc_pct=100 * r["acc"], gain_vs_tta_pp=100 * (r["acc"] - tta), gain_vs_base_pp=100 * (r["acc"] - base))
            per_base.append(r)
    aggregate = []
    for prefix, label in METHODS:
        group = [r for r in per_base if r["method"] == prefix]
        assert len(group) == 3
        r = dict(method=prefix, label=label, independent_bases=3)
        r.update({k: statistics.mean(x[k] for x in group) for k in ("acc_pct", "base_error_auroc", "ece", "nll", "gain_vs_tta_pp", "gain_vs_base_pp")})
        r["between_base_sd_gain_vs_tta_pp"] = statistics.stdev(x["gain_vs_tta_pp"] for x in group)
        aggregate.append(r)
    write(SOURCE / "per_base_methods.csv", per_base)
    write(SOURCE / "per_head_selected.csv", heads)
    write(SOURCE / "aggregate.csv", aggregate)
    early_heads = [r for r in heads if r["method"].startswith("base_resnet50__early_mean__")]
    audit = dict(
        independent_bases=3,
        heads_per_base=3,
        raw_prediction_replay_performed=False,
        resumed_pipeline_hours_excluding_data_and_prior_sessions=runtime,
        conditional_accuracy_vs_tta=intervals(early_heads, "ci_low_pp_vs_tta_selected", "ci_high_pp_vs_tta_selected"),
    )
    early = [r for r in per_base if r["method"] == "base_resnet50__early_mean"]
    for ref in ("best_augmented_mean", "history_mean", "best_margin"):
        group = [r for r in contrasts if r["reference"] == ref and r["encoder"] == "base_resnet50"]
        assert len(group) == 9
        audit[f"conditional_accuracy_vs_{ref}"] = intervals(group, "ci_low_pp", "ci_high_pp")
        audit[f"conditional_auroc_vs_{ref}"] = intervals(group, "auroc_ci_low", "auroc_ci_high")
        ref_rows = [r for r in per_base if r["method"] == "base_resnet50__" + ref]
        differences = [100 * (a["acc"] - b["acc"]) for a, b in zip(early, ref_rows)]
        audit[f"early_gain_vs_{ref}_pp"] = differences
        audit[f"early_mean_gain_vs_{ref}_pp"] = statistics.mean(differences)
    audit["early_gain_vs_tta_pp"] = [r["gain_vs_tta_pp"] for r in early]
    audit["early_mean_gain_vs_tta_pp"] = statistics.mean(audit["early_gain_vs_tta_pp"])
    audit["between_base_sample_sd_gain_vs_tta_pp"] = statistics.stdev(audit["early_gain_vs_tta_pp"])
    (SOURCE / "analysis_metrics.json").write_text(json.dumps(audit, indent=2) + "\n")

    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Method & Base42 & Base43 & Base44 & Mean & Error AUROC \\", r"\midrule"]
    for prefix, label in METHODS:
        values = [r["acc_pct"] for r in per_base if r["method"] == prefix]
        overall = next(r for r in aggregate if r["method"] == prefix)
        lines.append(f"{label} & " + " & ".join(f"{v:.2f}" for v in values) + f" & {overall['acc_pct']:.2f} & {overall['base_error_auroc']:.4f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (DEST / "tables/miniwebvision_replications.tex").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"early_mean": "#087f8c", "history_mean": "#9066ac", "best_augmented_mean": "#c48b32"}
    for target, color in colors.items():
        prefix = "base_resnet50__" + target
        values = [r for r in per_base if r["method"] == prefix]
        label = values[0]["label"]
        axes[0].plot([42, 43, 44], [r["gain_vs_tta_pp"] for r in values], "D-", color=color, label=label, markersize=5)
        axes[1].plot([42, 43, 44], [r["base_error_auroc"] for r in values], "D-", color=color, label=label, markersize=5)
        for seed in (42, 43, 44):
            group = [r for r in heads if r["base_seed"] == seed and r["method"].startswith(prefix + "__")]
            axes[0].scatter([seed] * 3, [float(r["gain_pp_vs_tta_selected"]) for r in group], color=color, alpha=0.4, s=17)
    axes[0].axhline(0, color="#777777", linestyle="--", linewidth=1, label="Uniform TTA")
    axes[1].plot([42, 43, 44], [r["base_error_auroc"] for r in per_base if r["method"] == "base"], "o--", color="#777777", label="MSP")
    for axis in axes:
        axis.set_xticks([42, 43, 44])
        axis.set_xlabel("Independent base seed")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy gain over selected TTA (pp)")
    axes[1].set_ylabel("Original-view base-error AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    labels[-1] = "Uniform TTA (left) / MSP (right)"
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.06), ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Mini-WebVision: three independent bases, three nested heads per base")
    fig.text(0.5, 0.01, "Diamonds: within-base means; small dots: heads. Lines connect base summaries; no confidence bands.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(DEST / f"figures/miniwebvision_replications.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
