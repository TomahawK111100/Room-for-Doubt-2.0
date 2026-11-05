"""Preserve and replay all thirteen frozen S2 controls; never fit or retune."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from omegaconf import OmegaConf
from room_for_doubt.data.posthoc_controls import (
    PINNED,
    POLICY_DIGESTS,
    REFERENCE_DIGESTS,
    mean_probabilities,
)
from room_for_doubt.data.temporal_evidence import probabilities
from room_for_doubt.analysis.posthoc_controls import branches
from room_for_doubt.evaluation.temporal_residual import (
    evaluate_probabilities,
    corrected_probabilities,
)
from room_for_doubt.evaluation.selection import paired_accuracy_interval
from room_for_doubt.evaluation.tie_aware_risk import expected_tie_aurc

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "1b6d50eadac07b6c3be0a371f230496315595955fcb94a6fbb9ebaa057ae66dd"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write_table(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def original(seed):
    root = (
        ROOT / "outputs/temporal_test_import_20261027/source_results"
        if seed == 42
        else ROOT / f"outputs/temporal_replications_import_20261027/source_seed{seed}"
    )
    name = (
        "temporal_residual_seed42_v1" if seed == 42 else f"temporal_base{seed}_head42"
    )
    path = root / "heads" / name / "final_evaluation/predictions.npz"
    if sha(path) != REFERENCE_DIGESTS[seed]:
        raise ValueError("Original reference differs")
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def analyze(sources, summaries, out):
    out = Path(out)
    if out.exists():
        raise ValueError("Fresh import directory required; preserve prior analysis")
    out.mkdir(parents=True)
    checks = []
    rows = []
    pairs = []
    fits = []
    stored = {}
    base_refs = {s: original(s) for s in (42, 43, 44)}

    def check(name, value):
        checks.append(dict(check=name, passed=bool(value)))
        if not value:
            raise ValueError(name)

    def close(x, y):
        return y is None if x is None else np.isclose(x, y, rtol=1e-8, atol=1e-10)

    def metrics(seed, name, reference, p):
        replay = evaluate_probabilities(reference["labels"], reference["base"], p)[
            "temporal_residual"
        ]
        replay["aurc_expected_ties"] = expected_tie_aurc(
            p.max(1), p.argmax(1) != reference["labels"]
        )
        rows.append(dict(seed=seed, method=name, **replay))
        stored[seed, name] = p
        return replay

    source_manifests = {}
    for package in ("A", "B"):
        dest = out / f"source_{package}"
        compact = out / f"summary_{package}"
        shutil.copytree(sources[package], dest)
        shutil.copytree(summaries[package], compact)
        fullfiles = [p for p in dest.rglob("*") if p.is_file()]
        source_manifests[package] = {
            str(p.relative_to(dest)): sha(p) for p in fullfiles
        }
        check(
            f"{package} source copy identical",
            all(
                sha(p) == sha(dest / p.relative_to(sources[package]))
                for p in Path(sources[package]).rglob("*")
                if p.is_file()
            ),
        )
        check(
            f"{package} compact identical to full",
            all(
                (dest / p.relative_to(compact)).is_file()
                and sha(p) == sha(dest / p.relative_to(compact))
                for p in compact.rglob("*")
                if p.is_file()
            ),
        )
        cfg = OmegaConf.to_container(OmegaConf.load(dest / "config.yaml"), resolve=True)
        check(
            f"{package} completed fixed run",
            read(dest / "metrics.json")["status"] == "accuracy_controls_complete"
            and cfg["package"] == package
            and cfg["benchmark"] is False
            and (
                cfg["head_epochs"],
                cfg["head_seed"],
                cfg["hidden_dim"],
                cfg["head_batch_size"],
                cfg["protocol"],
            )
            == (25, 42, 256, 256, "s2_v1"),
        )
        manifest = read(dest / "artifacts/source_manifest.json")
        launch = (
            ROOT
            / "outputs/posthoc_controls_launch_20261027_budget2h"
            / package
            / "source_snapshot.zip"
        )
        check(
            f"{package} original launch source archive",
            manifest["source_archive_sha256"] == SOURCE_SHA == sha(launch),
        )
        with zipfile.ZipFile(launch) as z:
            check(
                f"{package} every archived source file matches run",
                set(z.namelist()) == set(manifest["files"])
                and all(
                    hashlib.sha256(z.read(n)).hexdigest() == h
                    for n, h in manifest["files"].items()
                ),
            )
        group = read(dest / "all_policies_before_test.json")
        expected = {f"base{s}_{b}" for s, b in branches(package)}
        check(
            f"{package} exact complete package",
            set(group)
            == expected
            == {p.name for p in (dest / "heads").iterdir() if p.is_dir()},
        )
        witness = read(dest / "artifacts/canonical_witness.json")
        check(
            f"{package} recorded canonical validation witness",
            {r["seed"] for r in witness} == {42, 43, 44}
            and all(
                r["images"] == 64 and r["base_digest"] == PINNED[r["seed"]][0]
                for r in witness
            ),
        )
        check(
            f"{package} source preflight successful",
            read(dest / "preflight.json")["status"] == "verified_files",
        )
        for seed, branch in branches(package):
            key = f"base{seed}_{branch}"
            head = dest / "heads" / key
            policy_path = head / "artifacts/policy_before_test.json"
            p = read(policy_path)
            c = p["contract"]
            ref = base_refs[seed]
            check(
                key + " frozen policy matches group",
                sha(policy_path) == group[key]["policy_sha256"],
            )
            check(
                key + " declared source and target contract",
                c["recipient"] == seed
                and c["branch"] == branch
                and c["base_digest"] == PINNED[seed][0]
                and c["evidence_sha256"] == PINNED[seed][1]
                and c["head_seed"] == 42
                and c["parametrization"]
                == ("direct" if branch == "direct_soft5" else "residual")
                and p["test_accessed"] is False,
            )
            if branch == "independent2":
                donors = [d for d in (42, 43, 44) if d != seed]
                check(
                    key + " correct independent donors",
                    c["donors"] == donors
                    and c["donor_base_digests"]
                    == {str(d): PINNED[d][0] for d in donors},
                )
            fit = read(head / "metrics.json")
            history = list(csv.DictReader((head / "tables/training.csv").open()))
            vals = [fit["initial_meta_val_kl"]] + [
                float(r["meta_val_kl"]) for r in history
            ]
            check(
                key + " all25epochs and best meta checkpoint",
                [int(r["epoch"]) for r in history] == list(range(1, 26))
                and fit["best_epoch"] == int(np.argmin(vals))
                and close(min(vals), fit["best_meta_val_kl"]),
            )
            grid = list(
                csv.DictReader((head / "tables/validation_policies.csv").open())
            )
            chosen = max(
                grid,
                key=lambda r: (
                    float(r["val_noisy_acc"]),
                    -float(r["val_noisy_nll"]),
                    -float(r["alpha"]),
                ),
            )
            check(
                key + " alpha selection replay",
                float(chosen["alpha"]) == p["policy"]["alpha"]
                and close(float(chosen["val_noisy_acc"]), p["policy"]["val_noisy_acc"])
                and close(float(chosen["val_noisy_nll"]), p["policy"]["val_noisy_nll"]),
            )
            if branch != "supervised_head":
                grid2 = list(
                    csv.DictReader(
                        (head / "tables/validation_teacher_interpolation.csv").open()
                    )
                )
                chosen2 = max(
                    grid2,
                    key=lambda r: (
                        float(r["val_noisy_acc"]),
                        -float(r["val_noisy_nll"]),
                        -float(r["alpha"]),
                    ),
                )
                check(
                    key + " teacher interpolation replay",
                    float(chosen2["alpha"]) == p["teacher_policy"]["alpha"]
                    and close(
                        float(chosen2["val_noisy_acc"]),
                        p["teacher_policy"]["val_noisy_acc"],
                    ),
                )
            ev = head / "final_evaluation"
            m = read(ev / "metrics.json")
            ec = read(ev / "evaluation_contract.json")
            check(
                key + " evaluation identity",
                ec["policy_sha256"] == sha(policy_path)
                and ec["reference_sha256"] == REFERENCE_DIGESTS[seed]
                and sha(ev / "predictions.npz") == m["prediction_sha256"],
            )
            with np.load(ev / "predictions.npz", allow_pickle=False) as a:
                data = {k: a[k] for k in a.files}
            check(
                key + " original labels IDs and base match",
                all(np.array_equal(data[k], ref[k]) for k in ("ids", "labels", "base"))
                and len(ref["labels"]) == 10000
                and len(np.unique(ref["ids"])) == 10000,
            )
            for name, pred in data.items():
                if name in ("ids", "labels"):
                    continue
                probabilities(pred)
                metric = metrics(
                    seed,
                    branch if name == "corrected" else branch + "_" + name,
                    ref,
                    pred,
                )
                for k, v in metric.items():
                    if k == "aurc_expected_ties":
                        continue
                    check(key + " " + name + " " + k, close(v, m["methods"][name][k]))
            if branch == "independent2":
                check(
                    key + " independent teacher raw reconstruction",
                    np.array_equal(
                        data["teacher"],
                        mean_probabilities(
                            [base_refs[d]["base"] for d in (42, 43, 44) if d != seed]
                        ),
                    ),
                )
            if branch == "direct_soft5":
                check(
                    key + " original five-snapshot raw teacher",
                    np.array_equal(data["teacher"], ref["snapshot_mean"]),
                )
            if "teacher" in data:
                h = np.log(np.clip(data["teacher"], 1e-12, 1)) - np.log(
                    np.clip(data["base"], 1e-12, 1)
                )
                reconstructed = corrected_probabilities(
                    data["base"], h, p["teacher_policy"]["alpha"]
                )
                # Original CPU Torch ran on Kaggle x86; replay is on macOS ARM.
                # Keep saved raw values authoritative and disclose sub-micro FP32 drift.
                max_error = float(
                    np.max(np.abs(reconstructed - data["teacher_interpolation"]))
                )
                check(
                    key + " teacher mixture numerical reconstruction",
                    np.allclose(
                        reconstructed,
                        data["teacher_interpolation"],
                        rtol=2e-5,
                        atol=1e-6,
                    ),
                )
                check(
                    key + " teacher mixture same decisions",
                    np.array_equal(
                        reconstructed.argmax(1), data["teacher_interpolation"].argmax(1)
                    ),
                )
                checks.append(
                    dict(
                        check=key + " mixture maximum absolute FP32 replay error",
                        passed=True,
                        max_absolute_error=max_error,
                    )
                )

            delta = paired_accuracy_interval(
                data["labels"], data["base"].argmax(1), data["corrected"].argmax(1)
            )
            check(
                key + " base paired interval replay",
                all(close(v, m["paired_vs_base"][k]) for k, v in delta.items()),
            )
            before, after = (
                data["base"].argmax(1) == data["labels"],
                data["corrected"].argmax(1) == data["labels"],
            )
            check(
                key + " corrected harmed accounting",
                m["corrected"] == int((~before & after).sum())
                and m["harmed"] == int((before & ~after).sum())
                and m["changed"]
                == int((data["base"].argmax(1) != data["corrected"].argmax(1)).sum())
                and m["unchanged"] + m["changed"] == 10000
                and close(
                    (m["corrected"] - m["harmed"]) / 10000, delta["gain_pp"] / 100
                ),
            )
            fits.append(
                dict(
                    seed=seed,
                    branch=branch,
                    alpha=p["policy"]["alpha"],
                    best_epoch=fit["best_epoch"],
                    fit_seconds=fit["fit_seconds"],
                    selected_teacher_epoch=(
                        c["selected_teacher"]["epoch"]
                        if c["selected_teacher"]
                        else None
                    ),
                )
            )
    for seed, ref in base_refs.items():
        metrics(seed, "base", ref, ref["base"])
        metrics(seed, "early5", ref, ref["corrected"])
        metrics(seed, "early5_teacher", ref, ref["snapshot_mean"])
        if seed in (43, 44):
            path = (
                ROOT
                / f"outputs/temporal_selected_control_import_20261027/source_results/heads/selected_single_base{seed}_head42/final_evaluation/predictions.npz"
            )
            with np.load(path, allow_pickle=False) as a:
                check(
                    f"{seed} existing selected1 aligned",
                    all(
                        np.array_equal(a[k], ref[k]) for k in ("base", "labels", "ids")
                    ),
                )
                metrics(seed, "selected1", ref, a["corrected"])
                metrics(
                    seed,
                    "selected1_teacher_interpolation",
                    ref,
                    a["teacher_interpolation"],
                )
        comparisons = [
            ("early5", "selected1"),
            ("early5", "direct_soft5"),
            ("early5", "supervised_head"),
            ("early5", "early2"),
            ("early2", "independent2"),
            ("early5", "independent2"),
            ("early5", "selected1_teacher_interpolation"),
            ("early5", "independent2_teacher_interpolation"),
        ]
        for proposed, baseline in comparisons:
            paired = paired_accuracy_interval(
                ref["labels"],
                stored[seed, baseline].argmax(1),
                stored[seed, proposed].argmax(1),
            )
            pairs.append(
                dict(seed=seed, comparison=proposed + "__minus__" + baseline, **paired)
            )
    methods = [
        "base",
        "selected1",
        "supervised_head",
        "early2",
        "early5",
        "direct_soft5",
        "independent2",
        "early5_teacher",
        "selected1_teacher_interpolation",
        "independent2_teacher_interpolation",
    ]
    summary = []
    for method in methods:
        rr = [r for r in rows if r["method"] == method]
        check(method + " covers3bases", sorted(r["seed"] for r in rr) == [42, 43, 44])
        summary.append(
            dict(
                method=method,
                bases=3,
                acc_mean=float(np.mean([r["acc"] for r in rr])),
                acc_sample_sd=float(np.std([r["acc"] for r in rr], ddof=1)),
                nll_mean=float(np.mean([r["nll"] for r in rr])),
                ece_mean=float(np.mean([r["ece"] for r in rr])),
                aurc_mean=float(np.mean([r["aurc_expected_ties"] for r in rr])),
            )
        )
    for name, rr in [
        ("methods", rows),
        ("aggregate", summary),
        ("paired_comparisons", pairs),
        ("fit_settings", fits),
    ]:
        write_table(out / "tables" / f"{name}.csv", rr)
    stats = dict(
        status="raw_replay_complete",
        checks_passed=len(checks),
        new_heads=13,
        independent_bases=3,
        scope="post-pilot known-test reuse; descriptive across bases; conditional example intervals",
        runtime_benchmark="not_measured",
        novelty_acceptance=False,
    )
    for name, value in [
        ("checks", checks),
        ("metrics", stats),
        ("source_manifest", source_manifests),
        (
            "run_metadata",
            dict(
                source_dirs={k: str(v) for k, v in sources.items()},
                summary_dirs={k: str(v) for k, v in summaries.items()},
                source_archive_sha256=SOURCE_SHA,
                training_performed=False,
                raw_replay=True,
                scope_limits=[
                    "No imported head weights or train/val target arrays; no refit/forward regeneration",
                    "Selected42 full20-row teacher epoch grid omitted from small export; only chosen epoch and4-row interpolation grid available",
                    "AURC not exported by original runner; computed during local replay with expected ordering within ties",
                    "No independent hardware timing measurements",
                ],
            ),
        ),
    ]:
        (out / f"{name}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
    OmegaConf.save(
        OmegaConf.create(
            dict(
                run_id=out.name,
                stage="raw_replay",
                source_archive_sha256=SOURCE_SHA,
                packages=["A", "B"],
                head_seed=42,
                base_seeds=[42, 43, 44],
                conditional_bootstrap_draws=2000,
                conditional_bootstrap_seed=20261015,
            )
        ),
        out / "config.yaml",
    )
    (out / "report.md").write_text(
        "# Experiment Report: S2 post-hoc controls\n\n## Configuration\nSee config.yaml and preserved package configs.\n\n## Dataset\nOfficial10000-image CIFAR100 test; labels/IDs/base probabilities match prior frozen exports. Exploratory known-test reuse.\n\n## Checkpoint\nRecorded25-epoch histories, epoch0 and min meta-val KL selection replayed; weights not included in small imports.\n\n## Metrics\n"
        + json.dumps(stats, indent=2)
        + "\n\n## Compact Results\nSee tables/.\n\n## Generated Artifacts\nAll four preserved input folders, replay checks, manifests and full-precision tables.\n\n## Notes\nNo head fitting, target reconstruction from original training arrays or model forward performed. Independent donor raw test probabilities reconstructed. Teacher epoch selection42 cannot be fully replayed without omitted20-row grid. AURC was added during replay; runtime remains unmeasured.\n"
    )
    print(json.dumps(dict(**stats, aggregate=summary, pairs=pairs), indent=2))
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        default=str(ROOT / "outputs/posthoc_controls_import_20261028_verified"),
    )
    a = p.parse_args()
    downloads = Path("/Users/amirhansahapov/Downloads")
    analyze(
        {"A": downloads / "posthoc_results", "B": downloads / "posthoc_results-2"},
        {"A": downloads / "posthoc_summary", "B": downloads / "posthoc_summary-2"},
        a.output,
    )
