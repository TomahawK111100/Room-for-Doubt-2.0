"""Preserve follow-up exports and replay both branches without selecting on test."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from room_for_doubt.evaluation.temporal_residual import evaluate_probabilities
from room_for_doubt.evaluation.selection import paired_accuracy_interval


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def analyze(sources, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    methods = []
    comparisons = []
    runs = {}

    def check(name, value):
        checks.append(dict(check=name, passed=bool(value)))
        if not value:
            raise ValueError(name)

    for source in sources:
        source = Path(source)
        m = json.loads((source / "metrics.json").read_text())
        seed = m["base_seed"]
        check(
            f"{seed} distinct completed seed",
            seed not in runs and seed in (43, 44) and m["status"] == "complete",
        )
        dest = out / f"source_seed{seed}"
        if dest.exists():
            raise ValueError("Preserve prior import; choose fresh output")
        shutil.copytree(source, dest)
        check(
            f"{seed} source byte identical",
            all(
                sha(p) == sha(dest / p.relative_to(source))
                for p in source.rglob("*")
                if p.is_file()
            ),
        )
        frozen = json.loads((dest / "policies_before_any_test.json").read_text())
        protocol = json.loads((dest / "run_metadata.json").read_text())["protocol"]
        check(
            f"{seed} fixed split and head seed",
            protocol["base_split_seed"] == 42
            and protocol["head_seed"] == 42
            and protocol["seed"] == seed,
        )
        predictions = {}
        for branch, result in m["branches"].items():
            info = frozen[branch]
            head = dest / "heads" / info["run_id"]
            policy = head / "artifacts/policy_before_test.json"
            check(
                f"{seed} {branch} frozen policy bytes",
                sha(policy) == info["policy_sha256"],
            )
            p = json.loads(policy.read_text())
            check(
                f"{seed} {branch} policy selected before test",
                p["test_accessed"] is False,
            )
            expected = [1, 2, 3, 4, 5] if branch == "temporal" else [1]
            check(f"{seed} {branch} teacher epochs", p["teacher_epochs"] == expected)
            raw = np.load(head / "final_evaluation/predictions.npz", allow_pickle=False)
            d = {k: raw[k] for k in raw.files}
            raw.close()
            predictions[branch] = d
            check(
                f"{seed} {branch} official unique IDs",
                d["labels"].shape == (10000,) and len(np.unique(d["ids"])) == 10000,
            )
            for key, name in [
                ("base", "identity"),
                ("corrected", "temporal_residual"),
                ("snapshot_mean", f"snapshot_mean_{len(expected)}passes"),
                ("teacher_interpolation", "teacher_interpolation_extra_passes"),
            ]:
                replay = evaluate_probabilities(d["labels"], d["base"], d[key])[
                    "temporal_residual"
                ]
                for metric, value in replay.items():
                    saved = result["methods"][name][metric]
                    check(
                        f"{seed} {branch} {name} {metric}",
                        (
                            saved is None
                            if value is None
                            else np.isclose(value, saved, rtol=1e-8, atol=1e-10)
                        ),
                    )
                methods.append(dict(seed=seed, branch=branch, method=name, **replay))
            paired = paired_accuracy_interval(
                d["labels"], d["base"].argmax(1), d["corrected"].argmax(1)
            )
            for k, v in paired.items():
                check(
                    f"{seed} {branch} paired {k}",
                    np.isclose(v, result["paired_vs_identity"][k]),
                )
            comparisons.append(
                dict(seed=seed, comparison=branch + "_minus_base", **paired)
            )
        a, b = predictions["temporal"], predictions["single_epoch1"]
        check(
            f"{seed} matched branch inputs",
            all(np.array_equal(a[k], b[k]) for k in ("ids", "labels", "base")),
        )
        paired = paired_accuracy_interval(
            a["labels"], b["corrected"].argmax(1), a["corrected"].argmax(1)
        )
        for k, v in paired.items():
            check(
                f"{seed} branch paired {k}",
                np.isclose(v, m["temporal_minus_single_epoch1"][k]),
            )
        comparisons.append(
            dict(seed=seed, comparison="temporal_minus_single_epoch1", **paired)
        )
        runs[seed] = m
    check("two independent completed bases imported", set(runs) == {43, 44})
    prior = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "outputs/temporal_test_import_20261027/metrics.json"
        ).read_text()
    )
    for name, values in prior["methods"].items():
        methods.append(dict(seed=42, branch="temporal", method=name, **values))
    comparisons.append(
        dict(seed=42, comparison="temporal_minus_base", **prior["paired_vs_identity"])
    )
    aggregate = []
    for name in ["identity", "temporal_residual", "snapshot_mean_5passes"]:
        vals = [
            r["acc"]
            for r in methods
            if r["branch"] == "temporal" and r["method"] == name
        ]
        aggregate.append(
            dict(
                method=name,
                bases=3,
                acc_mean=float(np.mean(vals)),
                acc_sample_sd=float(np.std(vals, ddof=1)),
            )
        )
    gains = [
        r["gain_pp"] for r in comparisons if r["comparison"] == "temporal_minus_base"
    ]
    summary = dict(
        status="complete",
        checks_passed=len(checks),
        aggregate=aggregate,
        student_gain_mean_pp=float(np.mean(gains)),
        student_gain_sample_sd_pp=float(np.std(gains, ddof=1)),
        control_bases=2,
        temporal_minus_single_epoch1_mean_pp=float(
            np.mean(
                [
                    r["gain_pp"]
                    for r in comparisons
                    if r["comparison"] == "temporal_minus_single_epoch1"
                ]
            )
        ),
    )
    for n, rows in [
        ("methods", methods),
        ("paired_comparisons", comparisons),
        ("aggregate", aggregate),
    ]:
        write_rows(out / "tables" / f"{n}.csv", rows)
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    manifest = {
        str(p.relative_to(out)): sha(p)
        for p in out.glob("source_seed*/*")
        if p.is_file()
    }
    manifest = {
        str(p.relative_to(out)): sha(p)
        for d in out.glob("source_seed*")
        for p in d.rglob("*")
        if p.is_file()
    }
    (out / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "config.yaml").write_text(
        "run_id: temporal_replications_import_20261027\nseeds: [42,43,44]\ncontrol_seeds: [43,44]\ntraining_performed: false\n"
    )
    (out / "run_metadata.json").write_text(
        json.dumps(
            dict(
                sources=list(map(str, sources)),
                source_manifest_sha256=sha(out / "source_manifest.json"),
                raw_replay=True,
            ),
            indent=2,
        )
    )
    lines = [
        "# Experiment Report: temporal replications",
        "",
        "## Configuration",
        "Frozen follow-up protocol; head seed42 and fixed base split42.",
        "",
        "## Dataset",
        "10,000 official CIFAR100 test examples.",
        "",
        "## Checkpoint",
        "Base minimum noisy-validation CE; head minimum meta-validation KL; alpha chosen before test.",
        "",
        "## Metrics",
        json.dumps(summary, indent=2),
        "",
        "## Compact Results",
        "See tables/.",
        "",
        "## Generated Artifacts",
        "Preserved source folders, hashes, raw replay checks, tables.",
        "",
        "## Notes",
        "Three temporal bases; only two have the fixed epoch1 control. Paired intervals condition on fitted models, without multiplicity adjustment. Sample SD across three bases is descriptive.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("sources", nargs=2)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    analyze(a.sources, a.output)
