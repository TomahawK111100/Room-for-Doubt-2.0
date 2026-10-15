"""Factual compact reporting from saved architecture-study metrics."""

import csv
from collections import defaultdict
from pathlib import Path
import numpy as np


def write_study_summary(output_dir, config, metrics):
    out = Path(output_dir)
    groups = defaultdict(list)
    for row in metrics["results"]:
        groups[(row["dataset"], row["architecture"], row["score"])].append(row)
    aggregate = []
    for (dataset, architecture, score), rows in groups.items():
        values = dict(
            dataset=dataset,
            architecture=architecture,
            score=score,
            n_meta_seeds=len(rows),
        )
        for key in [
            "base_acc",
            "corrected_acc",
            "auroc",
            "auprc",
            "aurc",
            "tta_acc",
            "uw_tta_acc",
            "gain_pp",
            "uw_minus_tta_pp",
        ]:
            values[key + "_mean"] = float(np.mean([r[key] for r in rows]))
            values[key + "_sd"] = float(np.std([r[key] for r in rows], ddof=1))
        aggregate.append(values)
    with (out / "tables/aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    with (out / "report.md").open("w") as report:
        report.write(
            f"# Experiment Report: {config['run_id']}\n\n## Configuration\n\nResolved configuration: `config.yaml`. Full precision: `metrics.json`.\n"
        )
        report.write(
            f"\nSeeds: {list(config['seeds'])}. CFM: {config['flow_samples']} draws, {config['flow_steps']} Euler steps. Same frozen features, targets and partitions across architectures.\n"
        )
        report.write(
            "\n## Dataset\n\nCIFAR-10N Worse and CIFAR-100N noisy_label; 10000 test examples each. Historical complete-case trajectories: 15063 / 41541 usable source records. No new base training or encoder inference.\n"
        )
        report.write(
            "\n## Checkpoint\n\nAll models selected by minimum held-out reconstruction MSE. Old MLP-10 native MSE checkpoints are reused. Nine new heads were trained; total twelve architecture/dataset/seed checkpoints. Base checkpoints accepted by owner assumption; one base seed per dataset. Source paths: `run_metadata.json`.\n"
        )
        report.write(
            "\n## Metrics\n\nReconstruction is MSE of predicted mean trajectory against recorded noisy-class margins. Error metrics use the same base predictions. Top-2 and UW-TTA parameters are selected on noisy validation only, across all runs before test correctness. Mean-margin score is a fixed heuristic without clean-label detector fitting, not calibrated P(error). Oracle LDA is a separate reference.\n"
        )
        report.write(
            "\n## Compact Results\n\n### Common reconstruction and cost\n\n| Dataset | Architecture | Validation MSE | Trainable parameters | Time per 1000 trajectories, s |\n|---|---|---:|---:|---:|\n"
        )
        fidelity = defaultdict(list)
        for row in metrics["fidelity"]:
            fidelity[(row["dataset"], row["architecture"])].append(row)
        for (dataset, architecture), rows in fidelity.items():
            mse = np.mean([r["val_trajectory_mse"] for r in rows])
            cost = np.mean([1000 * r["seconds"] / r["samples"] for r in rows])
            report.write(
                f"| {dataset} | {architecture} | {mse:.4g} | {rows[0]['trainable_parameters']} | {cost:.4g} |\n"
            )
        report.write(
            f"\nValidation-only reconstruction preference: **{metrics['validation_reconstruction_preference']}**. Timing is a single-pass measurement including device overhead, not a sustained benchmark. Architectures have different capacities.\n"
        )
        for score, title in [
            ("mean_margin", "Primary score: no clean TRAIN detector supervision"),
            ("oracle_lda", "Oracle reference: clean TRAIN detector supervision"),
        ]:
            if not any(r["score"] == score for r in aggregate):
                continue
            report.write(
                f"\n### {title}\n\n| Dataset | Architecture | Base acc, % | Top-2 acc, % | Top-2 gain, pp | Error AUROC | Standard TTA, % | UW-TTA, % | UW minus TTA, pp |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
            )
            for r in aggregate:
                if r["score"] != score:
                    continue
                report.write(
                    f"| {r['dataset']} | {r['architecture']} | {100 * r['base_acc_mean']:.2f} | {100 * r['corrected_acc_mean']:.2f} | {r['gain_pp_mean']:.2f} | {r['auroc_mean']:.4f} | {100 * r['tta_acc_mean']:.2f} | {100 * r['uw_tta_acc_mean']:.2f} | {r['uw_minus_tta_pp_mean']:.2f} |\n"
                )
        report.write(
            "\n### Primary score: conditional paired 95% intervals\n\n| Dataset | Architecture | Seed | Top-2 gain, pp [95% CI] | UW minus TTA, pp [95% CI] |\n|---|---|---:|---:|---:|\n"
        )
        for row in metrics["results"]:
            if row["score"] != "mean_margin":
                continue
            report.write(
                f"| {row['dataset']} | {row['architecture']} | {row['seed']} | {row['gain_pp']:.2f} [{row['ci_low_pp']:.2f}, {row['ci_high_pp']:.2f}] | {row['uw_minus_tta_pp']:.2f} [{row['uw_ci_low_pp']:.2f}, {row['uw_ci_high_pp']:.2f}] |\n"
            )
        report.write(
            f"\n## Generated Artifacts\n\n- `metrics.json`: full precision fidelity, selections and all {len(metrics['results'])} results.\n- `tables/`: seed-level, aggregate, fidelity and noisy-validation selection tables.\n- `artifacts/`: validation scores/sweeps, all-selections marker and paired test predictions.\n- Selected head checkpoints remain in their source training directories.\n"
        )
        report.write(
            "\n## Notes\n\nThis is exploratory follow-up after prior test inspection, not untouched confirmation. Three meta initializations share one base model and split per dataset. Paired bootstrap conditions on the fitted models and validation choices. Native training losses differ; checkpoint comparison uses common reconstruction MSE. CFM scores the mean trajectory in both scoring arms.\n\nThe common cross-dataset action is original-view Top-2. Secondary TTA uses cached views: K=1 for CIFAR-10 (a no-op), K=20 for CIFAR-100. It cannot be interpreted as a controlled 1-versus-20-view architecture effect. Mean-margin is not a calibrated error probability. CIFAR-100 primary MLP badness AUROC remains below the previously measured MSP AUROC 0.8064. No universal performance, unseen-class vocabulary transfer, or Animal-10N/Mini-WebVision effectiveness is established.\n"
        )
    return aggregate
