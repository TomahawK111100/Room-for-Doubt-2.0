"""Compact aggregate tables and conditional paired accuracy intervals."""

from pathlib import Path
import csv
import json
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from room_for_doubt.evaluation.selection import paired_accuracy_interval
from room_for_doubt.reporting.artifacts import (
    generate_report,
    save_metrics,
    save_run_metadata,
)
from room_for_doubt.utils.paths import config_directory


@hydra.main(
    config_path=str(config_directory()),
    config_name="evaluation_summary",
    version_base=None,
)
def main(cfg):
    out = Path(HydraConfig.get().runtime.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    groups = {}
    intervals = []
    selected = []
    for input_dir in cfg.inputs:
        src = Path(to_absolute_path(input_dir))
        source_config = OmegaConf.load(src / "config.yaml")
        dataset = source_config.dataset.name
        metrics = json.loads((src / "metrics.json").read_text())
        with (src / "tables/summary.csv").open() as f:
            for row in csv.DictReader(f):
                key = (dataset, row["split"], row["method"])
                groups.setdefault(key, []).append(row)
        for seed, values in metrics["seeds"].items():
            selected.append(
                dict(dataset=dataset, seed=seed, selected=values["selected"])
            )
            with np.load(
                src / f"artifacts/test_seed{seed}.npz", allow_pickle=False
            ) as arrays:
                labels = arrays["labels"]
                if dataset == "cifar10n":
                    pairs = [
                        (
                            "top2_minus_base",
                            arrays["probabilities"].argmax(1),
                            arrays["corrected"],
                        )
                    ]
                else:
                    base = arrays["baseline_probabilities"].argmax(1)
                    standard = arrays["standard_probabilities"].argmax(1)
                    weighted = arrays["weighted_probabilities"].argmax(1)
                    pairs = [
                        ("uw_minus_base", base, weighted),
                        ("uw_minus_tta", standard, weighted),
                        ("tta_minus_base", base, standard),
                    ]
                for comparison, base, proposed in pairs:
                    intervals.append(
                        dict(
                            dataset=dataset,
                            seed=seed,
                            comparison=comparison,
                            **paired_accuracy_interval(labels, base, proposed),
                        )
                    )
    config = OmegaConf.to_container(cfg, resolve=True)
    metrics = dict(
        campaigns=len(cfg.inputs),
        meta_initializations_per_campaign=3,
        independent_base_seeds=1,
        test_examples_per_campaign=10000,
    )
    meta = dict(
        device="CPU reporting",
        checkpoint_path=None,
        checkpoint_assumption="owner confirmed correctness",
        supervision="oracle clean training labels",
    )
    save_metrics(out, metrics)
    save_run_metadata(out, cfg.run_id, config, **meta)
    generate_report(
        out,
        cfg.run_id,
        config,
        metrics,
        meta,
        "Tables aggregate three meta initializations with sample standard deviation (ddof=1). Baseline and all methods share one base checkpoint per dataset and the same examples. These are not independent base seeds. Paired bootstrap intervals use 2000 resamples of the 10000 test examples conditional on fitted models and chosen validation parameters; they do not cover base training, validation selection uncertainty or complete-case selection bias. Oracle detector supervision, shared learned class embedding, distinct noise/error targets and historical test reuse limit interpretation. No new method was selected from these test results. Aggregates are computed from full-precision metrics; displayed tables are rounded.",
    )
    fields = [
        "dataset",
        "split",
        "method",
        "n_meta_seeds",
        "acc_mean",
        "acc_sd",
        "auroc_mean",
        "auroc_sd",
        "auprc_mean",
        "aurc_mean",
        "ece_mean",
    ]
    aggregates = {}
    with (out / "tables/aggregate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (dataset, split, method), rows in groups.items():
            # Read full precision nested metrics, rather than the compact rounded CSV.
            src = next(
                Path(to_absolute_path(p))
                for p in cfg.inputs
                if OmegaConf.load(
                    Path(to_absolute_path(p)) / "config.yaml"
                ).dataset.name
                == dataset
            )
            full = json.loads((src / "metrics.json").read_text())["seeds"]
            entries = [
                v["ablation"][method]
                if split == "heldout_noise"
                else v["top2"]
                if method == "top2"
                else v["baseline"]
                if method == "msp"
                else v[method]
                for v in full.values()
            ]
            row = dict(
                dataset=dataset, split=split, method=method, n_meta_seeds=len(entries)
            )
            stats = {}
            for metric in ["acc", "auroc", "auprc", "aurc", "ece"]:
                values = [e[metric] for e in entries if metric in e]
                if values:
                    stats[metric + "_mean"] = float(np.mean(values))
                    row[metric + "_mean"] = format(np.mean(values), ".4g")
                    if metric in ["acc", "auroc"]:
                        stats[metric + "_sd"] = float(np.std(values, ddof=1))
                        row[metric + "_sd"] = format(np.std(values, ddof=1), ".4g")
            aggregates["/".join((dataset, split, method))] = stats
            w.writerow(row)
    for name, rows in [("bootstrap_ci", intervals), ("selection", selected)]:
        with (out / f"tables/{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        k: format(v, ".4g") if isinstance(v, float) else v
                        for k, v in r.items()
                    }
                )
    metrics.update(
        aggregates=aggregates, paired_intervals=intervals, selections=selected
    )
    save_metrics(out, metrics)
    # Scientific results are visible directly in the report, not only CSV files.
    with (out / "report.md").open("a") as report:
        report.write(
            "\n## Clean test accuracy\n\n| Dataset | Method | Accuracy, % | Meta SD, pp |\n|---|---|---:|---:|\n"
        )
        for key, values in aggregates.items():
            dataset, split, method = key.split("/")
            if (dataset == "cifar10n" and method in {"msp", "top2"}) or (
                dataset == "cifar100n" and method in {"msp", "tta", "uw_tta"}
            ):
                report.write(
                    f"| {dataset} | {method} | {100 * values['acc_mean']:.2f} | {100 * values['acc_sd']:.2f} |\n"
                )
        report.write(
            "\n## Oracle matched noise-label detection\n\n| Dataset | Representation | AUROC mean | Meta SD |\n|---|---|---:|---:|\n"
        )
        for key, values in aggregates.items():
            dataset, split, method = key.split("/")
            if split == "heldout_noise":
                report.write(
                    f"| {dataset} | {method} | {values['auroc_mean']:.4f} | {values['auroc_sd']:.4f} |\n"
                )
        report.write(
            "\n## Error detection on the same base predictions\n\n| Dataset | Score | AUROC mean | Meta SD |\n|---|---|---:|---:|\n"
        )
        for key, values in aggregates.items():
            dataset, split, method = key.split("/")
            if split == "test_error" and method in {
                "msp",
                "entropy",
                "margin",
                "badness",
            }:
                report.write(
                    f"| {dataset} | {method} | {values['auroc_mean']:.4f} | {values['auroc_sd']:.4f} |\n"
                )
        report.write(
            "\n## Paired conditional 95% accuracy intervals\n\n| Dataset | Seed | Comparison | Gain, pp | 95% interval, pp |\n|---|---:|---|---:|---:|\n"
        )
        for row in intervals:
            report.write(
                f"| {row['dataset']} | {row['seed']} | {row['comparison']} | {row['gain_pp']:.2f} | [{row['ci_low_pp']:.2f}, {row['ci_high_pp']:.2f}] |\n"
            )
        report.write(
            "\nAll three UW-TTA versus TTA intervals include or touch zero. These intervals condition on the fitted models and validation selection; they do not measure independent base-training variation.\n"
        )
    print(out / "report.md")


if __name__ == "__main__":
    main()
