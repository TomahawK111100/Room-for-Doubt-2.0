"""Preserve frozen-test exports and replay metrics from saved probabilities."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from room_for_doubt.evaluation.temporal_residual import evaluate_probabilities
from room_for_doubt.evaluation.selection import paired_accuracy_interval

HEAD = "temporal_residual_seed42_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(source, output):
    source, out = Path(source).resolve(), Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    preserved = out / "source_results"
    if preserved.exists():
        raise ValueError("Preserve the prior import; choose a new output")
    if source.is_dir():
        shutil.copytree(source, preserved)
    else:
        shutil.copy2(source, out / source.name)
        with zipfile.ZipFile(source) as z:
            for name in z.namelist():
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Unsafe archive member")
            z.extractall(preserved)
    manifest = {
        str(p.relative_to(preserved)): digest(p)
        for p in preserved.rglob("*")
        if p.is_file()
    }
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    metrics = json.loads((preserved / "metrics.json").read_text())
    metadata = json.loads((preserved / "run_metadata.json").read_text())
    raw = preserved / "heads" / HEAD / "final_evaluation/predictions.npz"
    checks = []

    def check(name, passed):
        checks.append(dict(check=name, passed=bool(passed)))
        if not passed:
            raise ValueError(name)

    check(
        "prediction SHA256 matches metadata",
        digest(raw) == metadata["prediction_sha256"],
    )
    check(
        "policy SHA256 matches metadata",
        digest(preserved / "policy_before_test.json") == metadata["policy_sha256"],
    )
    check(
        "evaluation completed without training",
        metrics["status"] == "complete"
        and metadata["training_performed"] is False
        and metadata["policy_changed"] is False,
    )
    with np.load(raw, allow_pickle=False) as d:
        labels, base = d["labels"], d["base"]
        check(
            "official test cardinality",
            labels.shape == (10000,) and base.shape == (10000, 100),
        )
        check("unique test IDs", len(np.unique(d["ids"])) == 10000)
        rows = {}
        for key, name in (
            ("base", "identity"),
            ("corrected", "temporal_residual"),
            ("snapshot_mean", "snapshot_mean_5passes"),
            ("teacher_interpolation", "teacher_interpolation_extra_passes"),
        ):
            p = d[key]
            check(
                name + " valid probabilities",
                np.isfinite(p).all()
                and (p >= 0).all()
                and np.allclose(p.sum(1), 1, atol=2e-5),
            )
            replay = evaluate_probabilities(labels, base, p)["temporal_residual"]
            for metric, value in replay.items():
                saved = metrics["methods"][name][metric]
                check(
                    name + " replay " + metric,
                    (
                        saved is None
                        if value is None
                        else np.isclose(value, saved, rtol=1e-8, atol=1e-10)
                    ),
                )
            rows[name] = replay
        pairs = []
        for saved in metrics["paired_comparisons"]:
            replay = paired_accuracy_interval(
                labels, d[saved["baseline"]].argmax(1), d[saved["proposed"]].argmax(1)
            )
            for k, v in replay.items():
                check(saved["proposed"] + " paired " + k, np.isclose(v, saved[k]))
            pairs.append(saved)
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    shutil.copy2(preserved / "config.yaml", out / "config.yaml")
    (out / "run_metadata.json").write_text(
        json.dumps(
            dict(
                source=str(source),
                source_manifest_sha256=digest(out / "source_manifest.json"),
                raw_replay=True,
                checks_passed=len(checks),
            ),
            indent=2,
        )
    )
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    for filename, values in (
        ("methods.csv", [dict(method=k, **v) for k, v in rows.items()]),
        ("paired_comparisons.csv", pairs),
    ):
        with (tables / filename).open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(values[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(values)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["identity", "temporal_residual", "snapshot_mean_5passes"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].bar(
        ["Base", "Student", "5-snapshot teacher"],
        [100 * rows[n]["acc"] for n in names],
        color=["#777777", "#277c8e", "#c49337"],
    )
    axes[0].set_ylabel("Clean-test accuracy (%)")
    axes[0].set_ylim(
        min(100 * rows[n]["acc"] for n in names) - 2,
        max(100 * rows[n]["acc"] for n in names) + 1,
    )
    for i, n in enumerate(names):
        axes[0].text(
            i, 100 * rows[n]["acc"] + 0.1, f"{100*rows[n]['acc']:.2f}", ha="center"
        )
    for i, pair in enumerate(pairs):
        axes[1].errorbar(
            pair["gain_pp"],
            i,
            xerr=[
                [pair["gain_pp"] - pair["ci_low_pp"]],
                [pair["ci_high_pp"] - pair["gain_pp"]],
            ],
            fmt="o",
            color="#277c8e",
            capsize=4,
        )
    axes[1].set_yticks(
        range(3), ["Student − base", "Teacher − base", "Student − teacher"]
    )
    axes[1].axvline(0, color="gray", lw=1)
    axes[1].set_xlabel("Accuracy difference (pp), nominal 95% paired CI", fontsize=8)
    fig.suptitle("Frozen CIFAR-100N temporal pilot: one fitted base")
    for suffix in ("png", "pdf"):
        fig.savefig(plots / ("frozen_test." + suffix), dpi=180)
    plt.close(fig)
    lines = [
        "# Experiment Report: frozen temporal clean test",
        "",
        "## Configuration",
        "",
        "Evaluation only; frozen alpha=1. No training or test-based selection.",
        "",
        "## Dataset",
        "",
        "10,000 official CIFAR-100 clean test images.",
        "",
        "## Checkpoint",
        "",
        "Frozen pilot identities are preserved in source_results/run_metadata.json.",
        "",
        "## Metrics",
        "",
    ]
    lines += [
        f"- {n}: accuracy={100*rows[n]['acc']:.2f}%, NLL={rows[n]['nll']:.4f}."
        for n in names
    ]
    lines += ["", "## Compact Results", ""] + [
        f"- {p['proposed']} minus {p['baseline']}: {p['gain_pp']:+.2f} pp, nominal interval [{p['ci_low_pp']:+.2f}, {p['ci_high_pp']:+.2f}]."
        for p in pairs
    ]
    lines += [
        "",
        "## Generated Artifacts",
        "",
        "Raw source_results, hash manifest, replay checks, tables, plots.",
        "",
        "## Notes",
        "",
        f"{len(checks)} checks passed. Intervals condition on one fitted base/head and have no multiplicity adjustment. Independent-seed replication and matched non-temporal controls remain outstanding.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(methods=rows, paired=pairs, checks=len(checks)), indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.output)
