"""Import and check a compact validation-only pilot; never infer test accuracy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("bases/temporal_cifar100n_seed42_v1")
HEAD = Path("heads/temporal_residual_seed42_v1")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def analyze(source, output):
    source, out = Path(source).resolve(), Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    imported = out / "source_summary"
    if not imported.exists():
        shutil.copytree(source, imported)
    files = sorted(p for p in source.rglob("*") if p.is_file())
    original = {str(p.relative_to(source)): digest(p) for p in files}
    copied = {
        str(p.relative_to(imported)): digest(p)
        for p in imported.rglob("*")
        if p.is_file()
    }
    if original != copied:
        raise ValueError(
            "Source and preserved import differ; never overwrite prior evidence"
        )
    (out / "source_manifest.json").write_text(json.dumps(original, indent=2))
    r = imported
    top = read_json(r / "metrics.json")
    base = read_json(r / BASE / "metrics.json")
    head = read_json(r / HEAD / "metrics.json")
    policy = read_json(r / HEAD / "artifacts/policy_before_test.json")
    metadata = read_json(r / HEAD / "run_metadata.json")
    prepare = read_json(r / HEAD / "prepare/run_metadata.json")
    prepare_metrics = read_json(r / HEAD / "prepare/metrics.json")
    base_cfg = OmegaConf.load(r / BASE / "config.yaml")
    head_cfg = OmegaConf.load(r / HEAD / "config.yaml")
    history = rows(r / BASE / "history.csv")
    train = rows(r / HEAD / "tables/training.csv")
    student_grid = rows(r / HEAD / "tables/validation_policies.csv")
    teacher_grid = rows(r / HEAD / "tables/validation_teacher_interpolation.csv")
    source_code = read_json(r / "artifacts/source_manifest.json")
    launch = read_json(
        ROOT / "outputs/temporal_pilot_launch_20261026/source_manifest.json"
    )
    witness = read_json(r / "artifacts/kernel_witness.json")
    checks = []

    def check(name, condition):
        checks.append(dict(check=name, passed=bool(condition)))
        if not condition:
            raise ValueError(f"Compact evidence check failed: {name}")

    check("all source files preserved byte for byte", original == copied)
    check(
        "all three stages report complete/frozen",
        top["status"] == "validation_complete_test_deferred"
        and base["status"] == "complete"
        and head["status"] == "policy_frozen",
    )
    check("top-level and head metrics match", top["head_metrics"] == head)
    check(
        "recorded clean test deferred",
        top["test_evaluated"] is False
        and base["test_evaluated"] is False
        and head["test_evaluated"] is False
        and policy["test_accessed"] is False
        and base_cfg.defer_test_evaluation,
    )
    check(
        "fixed base dataset and seed",
        base_cfg.dataset.name == "cifar100n"
        and base_cfg.dataset.noisy_split == "noisy100"
        and base_cfg.seed == 42
        and base_cfg.base_split_seed == 42,
    )
    check(
        "complete ordered 50-epoch base history",
        [int(x["epoch"]) for x in history] == list(range(50))
        and base["completed_epochs"] == 50,
    )
    best_base = min(history, key=lambda x: float(x["val_loss"]))
    check(
        "base checkpoint metadata matches noisy-val minimum",
        int(best_base["epoch"]) == base["best_epoch"]
        and np.isclose(
            float(best_base["val_loss"]), base["best_val_loss"], atol=1e-10, rtol=0
        ),
    )
    check(
        "complete ordered 25-epoch head history",
        [int(x["epoch"]) for x in train] == list(range(1, 26))
        and head_cfg.epochs == 25,
    )
    best_head = min(train, key=lambda x: float(x["meta_val_kl"]))
    check(
        "head checkpoint metadata matches KL minimum",
        int(best_head["epoch"]) == head["best_epoch"]
        and np.isclose(
            float(best_head["meta_val_kl"]),
            head["best_meta_val_kl"],
            atol=1e-12,
            rtol=0,
        )
        and head["best_meta_val_kl"] <= head["initial_meta_val_kl"],
    )
    check(
        "head/checkpoint feature dimensions",
        policy["classes"] == 100
        and policy["feature_width"] == 2048
        and policy["hidden_dim"] == head_cfg.hidden_dim == 256,
    )
    check(
        "teacher epochs fixed before evaluation",
        policy["teacher_epochs"]
        == list(base_cfg.temporal_snapshot_epochs)
        == list(head_cfg.teacher_epochs)
        == [1, 2, 3, 4, 5],
    )
    check(
        "base identity agrees in compact records",
        policy["base_digest"] == metadata["base_digest"] == prepare["base_digest"],
    )
    check(
        "cache identity agrees in compact records",
        policy["cache_manifest_sha256"]
        == metadata["cache_manifest_sha256"]
        == prepare["cache_manifest_sha256"],
    )
    check(
        "frozen policy bytes agree with metadata hash",
        digest(r / HEAD / "artifacts/policy_before_test.json")
        == metadata["policy_sha256"],
    )
    check(
        "source code agrees with submitted snapshot",
        source_code == launch
        and source_code["source_archive_sha256"] == witness["source_sha256"],
    )
    check(
        "two GPU witness records finite",
        len(witness["gpus"]) == 2
        and all(g["finite"] and g["name"] == "Tesla T4" for g in witness["gpus"]),
    )
    check(
        "prepare reports original split counts",
        prepare_metrics["train_images"] == 45000
        and prepare_metrics["noisy_val_images"] == 5000
        and prepare_metrics["test_accessed"] is False,
    )

    def selected(grid, key):
        check(
            key + " fixed alpha grid",
            [float(x["alpha"]) for x in grid] == list(head_cfg.alphas),
        )
        for row in grid:
            check(
                key + " finite valid row alpha=" + row["alpha"],
                np.isfinite([float(v) for v in row.values()]).all()
                and 0 <= float(row["val_noisy_acc"]) <= 1,
            )
        chosen = max(
            grid,
            key=lambda x: (
                float(x["val_noisy_acc"]),
                -float(x["val_noisy_nll"]),
                -float(x["alpha"]),
            ),
        )
        check(
            key + " frozen selection matches full saved grid",
            all(
                np.isclose(float(chosen[k]), policy[key][k], rtol=0, atol=1e-12)
                for k in chosen
            ),
        )
        return chosen

    student, teacher = selected(student_grid, "policy"), selected(
        teacher_grid, "teacher_policy"
    )
    identity = student_grid[0]
    check(
        "both identity rows agree",
        identity == teacher_grid[0]
        and np.isclose(
            float(identity["val_noisy_acc"]), base["val_noisy_acc"], rtol=0, atol=1e-12
        ),
    )
    for chosen, prefix in ((student, "student"), (teacher, "teacher")):
        gain = 100 * (float(chosen["val_noisy_acc"]) - float(identity["val_noisy_acc"]))
        check(
            prefix + " reported gain agrees",
            np.isclose(gain, head[prefix + "_val_gain_pp"], rtol=0, atol=1e-10),
        )
    check(
        "selected student alpha matches metrics",
        float(student["alpha"]) == head["selected_alpha"],
    )
    check(
        "selected teacher alpha matches metrics",
        float(teacher["alpha"]) == head["teacher_selected_alpha"],
    )
    check(
        "selected accuracies match metrics",
        float(student["val_noisy_acc"]) == head["val_noisy_acc"]
        and float(teacher["val_noisy_acc"]) == head["teacher_val_noisy_acc"],
    )

    methods = [
        dict(
            method=name,
            val_noisy_acc_pct=100 * float(row["val_noisy_acc"]),
            val_noisy_nll=float(row["val_noisy_nll"]),
            gain_pp=100
            * (float(row["val_noisy_acc"]) - float(identity["val_noisy_acc"])),
        )
        for name, row in (
            ("base_fp32", identity),
            ("residual_student", student),
            ("snapshot_mean_5", teacher),
        )
    ]
    (out / "tables").mkdir(exist_ok=True)
    with (out / "tables/methods.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(methods[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {
                k: float(f"{v:.4g}") if isinstance(v, float) else v
                for k, v in row.items()
            }
            for row in methods
        )
    metrics = dict(
        status="validation_pilot_complete_test_pending",
        methods=methods,
        checks_passed=len(checks),
        source_files=len(files),
        independent_bases=1,
        head_seeds=1,
        student_teacher_gain_ratio=head["student_val_gain_pp"]
        / head["teacher_val_gain_pp"],
        base_epoch_seconds=sum(float(x["seconds"]) for x in history),
        head_best_epoch=head["best_epoch"],
        base_best_epoch_one_based=base["best_epoch"] + 1,
        meta_validation_kl_reduction_pct=100
        * (1 - head["best_meta_val_kl"] / head["initial_meta_val_kl"]),
        clean_test_evaluated=False,
        raw_weights_verified=False,
        raw_probabilities_verified=False,
        conditional_intervals_recomputed=False,
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    OmegaConf.save(
        OmegaConf.create(
            dict(
                run_id=out.name,
                dataset=OmegaConf.to_container(base_cfg.dataset),
                seed=42,
                input_summary=str(source),
                analysis="compact validation checks; no refit or test selection",
            )
        ),
        out / "config.yaml",
    )
    (out / "run_metadata.json").write_text(
        json.dumps(
            dict(
                source_manifest_sha256=digest(out / "source_manifest.json"),
                base_digest=policy["base_digest"],
                head_digest=policy["checkpoint_sha256"],
                policy_sha256=metadata["policy_sha256"],
                code_archive_sha256=source_code["source_archive_sha256"],
                verification_scope="compact only; binaries and per-example arrays absent",
            ),
            indent=2,
        )
    )
    progress = read_json(r / "progress.json")
    note = (
        "The ZIP contains a stale progress.json captured at fit/running. Completed root/head metrics and frozen policy agree. Original bytes were retained; no source status was rewritten."
        if progress["status"] == "running"
        else "No stale progress marker detected."
    )
    report = f"""# Experiment Report: temporal residual validation pilot

## Configuration
One CIFAR100N base seed42,50epochs; one residual head seed42,25epochs. Fixed alpha grid0/.25/.5/1.

## Dataset
45,000 base-train and5,000 noisy-validation images. All accuracies below use observed noisy labels. Clean test not evaluated.

## Checkpoint
Base selected at epoch{base['best_epoch'] + 1} (one-based); head at epoch{head['best_epoch']}. Both selected minima match saved histories. Binary weights are absent from the compact summary.

## Metrics
{len(checks)} compact consistency checks pass. Source copy preserves{len(files)} files byte for byte.

## Compact Results

| Method | Noisy-val accuracy | Gain vs base | Noisy-val NLL |
|---|---:|---:|---:|
"""
    for row in methods:
        report += f"| {row['method']} | {row['val_noisy_acc_pct']:.4g}% | {row['gain_pp']:+.4g}pp | {row['val_noisy_nll']:.4g} |\n"
    report += f"""
## Generated Artifacts
source_summary/, source_manifest.json, checks.json, metrics.json, tables/methods.csv, plots/validation_pilot.png/pdf.

## Notes
{note}

Recorded base epoch time sums to{metrics['base_epoch_seconds'] / 3600:.3f}hours, excluding setup/data and downstream work. Student retains{100 * metrics['student_teacher_gain_ratio']:.3g}%of the selected teacher validation gain; this is a descriptive single-split ratio, not a transfer guarantee. Original base CE is AMP, whereas identity row NLL is FP32. No significance interval, clean-test improvement, cross-base replication or temporal-attribution claim can be reconstructed from these compact tables.
"""
    (out / "report.md").write_text(report)
    return metrics, student_grid, teacher_grid, train


def plot(out, metrics, student_grid, teacher_grid, train):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out) / "plots"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#707b88", "#2371aa", "#d68c22"]
    values = [r["val_noisy_acc_pct"] for r in metrics["methods"]]
    axes[0].scatter(range(3), values, color=colors, s=90, zorder=3)
    axes[0].set_xticks(range(3), ["Base", "Student", "5-snapshot\nteacher"])
    axes[0].set_xlim(-.5, 2.5)
    axes[0].set_ylim(54, 57.2)
    axes[0].set_ylabel("Noisy-validation accuracy (%)")
    for i, value in enumerate(values):
        axes[0].text(i, value + 0.13, f"{value:.2f}", ha="center")
    axes[0].set_title("One base, one head seed")
    epochs = [int(r["epoch"]) for r in train]
    axes[1].plot(epochs, [float(r["train_kl"]) for r in train], label="Meta-train")
    axes[1].plot(
        epochs, [float(r["meta_val_kl"]) for r in train], label="Meta-validation"
    )
    axes[1].axvline(
        metrics["head_best_epoch"], color="grey", linestyle="--", label="Selected epoch"
    )
    axes[1].set(
        xlabel="Head epoch", ylabel="Teacher KL", title="Checkpoint selected at epoch4"
    )
    axes[1].legend(frameon=False, fontsize=8)
    for grid, label, color in (
        (student_grid, "Student", colors[1]),
        (teacher_grid, "Teacher/base mixture", colors[2]),
    ):
        axes[2].plot(
            [float(r["alpha"]) for r in grid],
            [100 * float(r["val_noisy_acc"]) for r in grid],
            "o-",
            color=color,
            label=label,
        )
    axes[2].set(
        xlabel="Fixed correction strength alpha",
        ylabel="Noisy-validation accuracy (%)",
        title="Saved selection grid",
    )
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CIFAR-100N temporal compression pilot — validation only", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "5,000 noisy-validation labels; no clean-test result or significance interval",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"validation_pilot.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--output", default=str(ROOT / "outputs/temporal_pilot_import_20261027")
    )
    args = parser.parse_args()
    result = analyze(args.source, args.output)
    plot(args.output, *result)
    print(json.dumps(result[0], indent=2))
