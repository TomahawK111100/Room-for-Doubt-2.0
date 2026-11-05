"""Replay exploratory selected-single controls against preserved temporal results."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from room_for_doubt.evaluation.temporal_residual import evaluate_probabilities
from room_for_doubt.evaluation.selection import paired_accuracy_interval

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def analyze(source, summary, out):
    source, summary, out = map(Path, (source, summary, out))
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "source_results"
    if dest.exists():
        raise ValueError("Preserve previous import")
    shutil.copytree(source, dest)
    m = json.loads((dest / "metrics.json").read_text())
    frozen = json.loads((dest / "all_policies_before_test.json").read_text())
    checks = []
    rows = []
    pairs = []

    def check(n, v):
        checks.append(dict(check=n, passed=bool(v)))
        if not v:
            raise ValueError(n)

    check(
        "completed exploratory controls",
        m["status"] == "complete"
        and m["protocol"]["base_training"] is False
        and m["protocol"]["seeds"] == [43, 44],
    )
    check(
        "source copy byte identical",
        all(
            sha(p) == sha(dest / p.relative_to(source))
            for p in source.rglob("*")
            if p.is_file()
        ),
    )
    compact = [p for p in summary.rglob("*") if p.is_file()]
    check(
        "all compact files match full export",
        all(
            (dest / p.relative_to(summary)).is_file()
            and sha(p) == sha(dest / p.relative_to(summary))
            for p in compact
        ),
    )
    for seed in (43, 44):
        result = m["results"][str(seed)]
        f = frozen[str(seed)]
        head = dest / "heads" / f"selected_single_base{seed}_head42"
        policy = head / "artifacts/policy_before_test.json"
        p = json.loads(policy.read_text())
        check(f"{seed} frozen policy hash", sha(policy) == f["policy_sha256"])
        check(f"{seed} policy precedes evaluation", p["test_accessed"] is False)
        grid = list(csv.DictReader((dest / f"selection_seed{seed}.csv").open()))
        chosen = max(
            grid,
            key=lambda r: (
                float(r["acc"]),
                -float(r["nll"]),
                -float(r["alpha"]),
                -int(r["epoch"]),
            ),
        )
        check(
            f"{seed} validation-only epoch selection replay",
            int(chosen["epoch"]) == result["selected_epoch"] == p["teacher_epochs"][0],
        )
        with np.load(
            head / "final_evaluation/predictions.npz", allow_pickle=False
        ) as data:
            labels, base = data["labels"], data["base"]
            check(
                f"{seed} full unique test IDs",
                len(labels) == 10000 and len(np.unique(data["ids"])) == 10000,
            )
            for key, name in [
                ("base", "identity"),
                ("corrected", "temporal_residual"),
                ("snapshot_mean", "snapshot_mean_1passes"),
                ("teacher_interpolation", "teacher_interpolation_extra_passes"),
            ]:
                replay = evaluate_probabilities(labels, base, data[key])[
                    "temporal_residual"
                ]
                for k, v in replay.items():
                    check(
                        f"{seed} {name} {k}",
                        (
                            result["control"]["methods"][name][k] is None
                            if v is None
                            else np.isclose(
                                v,
                                result["control"]["methods"][name][k],
                                rtol=1e-8,
                                atol=1e-10,
                            )
                        ),
                    )
                rows.append(dict(seed=seed, method="selected_" + name, **replay))
            paired = paired_accuracy_interval(
                labels, base.argmax(1), data["corrected"].argmax(1)
            )
            for k, v in paired.items():
                check(
                    f"{seed} control vs base {k}",
                    np.isclose(v, result["control"]["paired_vs_identity"][k]),
                )
            pairs.append(
                dict(seed=seed, comparison="selected_student_minus_base", **paired)
            )
            original = (
                ROOT
                / f"outputs/temporal_replications_import_20261027/source_seed{seed}/heads/temporal_base{seed}_head42/final_evaluation/predictions.npz"
            )
            with np.load(original, allow_pickle=False) as a:
                check(
                    f"{seed} matched base labels IDs",
                    all(
                        np.array_equal(a[k], data[k]) for k in ("base", "labels", "ids")
                    ),
                )
                paired = paired_accuracy_interval(
                    labels, data["corrected"].argmax(1), a["corrected"].argmax(1)
                )
                for k, v in paired.items():
                    check(
                        f"{seed} temporal vs selected {k}",
                        np.isclose(v, result["temporal_minus_selected_single"][k]),
                    )
                pairs.append(
                    dict(
                        seed=seed,
                        comparison="temporal_minus_selected_student",
                        **paired,
                    )
                )
                pairs.append(
                    dict(
                        seed=seed,
                        comparison="temporal_minus_selected_teacher_interpolation",
                        **paired_accuracy_interval(
                            labels,
                            data["teacher_interpolation"].argmax(1),
                            a["corrected"].argmax(1),
                        ),
                    )
                )
                replay = evaluate_probabilities(labels, base, a["corrected"])[
                    "temporal_residual"
                ]
                rows.append(dict(seed=seed, method="temporal_student", **replay))
    stats = dict(
        status="complete",
        checks_passed=len(checks),
        mean_temporal_minus_selected_pp=float(
            np.mean(
                [
                    r["gain_pp"]
                    for r in pairs
                    if r["comparison"] == "temporal_minus_selected_student"
                ]
            )
        ),
        scope=m["protocol"]["test_scope"],
    )
    for name, values in [("methods", rows), ("paired_comparisons", pairs)]:
        pth = out / "tables" / f"{name}.csv"
        pth.parent.mkdir(exist_ok=True)
        with pth.open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(values[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(values)
    for name, values in [
        ("checks", checks),
        ("metrics", stats),
        (
            "source_manifest",
            {str(p.relative_to(dest)): sha(p) for p in dest.rglob("*") if p.is_file()},
        ),
        (
            "run_metadata",
            dict(
                source=str(source),
                summary=str(summary),
                training_performed=False,
                raw_replay=True,
            ),
        ),
    ]:
        (out / f"{name}.json").write_text(json.dumps(values, indent=2))
    shutil.copy2(dest / "config.yaml", out / "config.yaml")
    (out / "report.md").write_text(
        "# Experiment Report: selected single-snapshot control\n\n## Configuration\nOriginal fixed follow-up protocol; no base retraining.\n\n## Dataset\nKnown CIFAR100 test, post-pilot exploratory reuse.\n\n## Checkpoint\nSelected teacher by noisy validation, head by meta-validation KL.\n\n## Metrics\n"
        + json.dumps(stats, indent=2)
        + "\n\n## Compact Results\nSee tables/.\n\n## Generated Artifacts\nOriginal source copy, SHA256 manifest, replay checks and tables.\n\n## Notes\nConditional paired intervals, no multiplicity adjustment, two base seeds. Teacher interpolation requires an additional backbone pass.\n"
    )
    print(json.dumps(dict(**stats, pairs=pairs), indent=2))
    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("summary")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    analyze(a.source, a.summary, a.output)
