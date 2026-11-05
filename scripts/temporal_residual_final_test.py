"""Evaluate the frozen pilot using its own archived code, without fitting anything."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

EXPECTED = dict(
    source_archive_sha256="2523c77e20ce00178e28eced2f1b9910b2e8e7c808c4f2ac1fd34cb1614acd9c",
    base_digest="528f45f2d36efabed062716b0f4e6690d396ca4370dba26497cb4ea5ebff0a6a",
    checkpoint_sha256="ed159cfa8ba0a9ee3f421851e14044b7bad03be132befdbe12e20c3c5a3bdc05",
    cache_manifest_sha256="1c45838acbdd07d101195e4a51e11e47982bced8c02282b3b9242d8538c89aad",
    policy_sha256="e1dd54ce3d1fd630d8b7fb7d67f9f9f92fdfad4e1854787332b1dfef47354f57",
)
HEAD = "temporal_residual_seed42_v1"
BASE = "temporal_cifar100n_seed42_v1"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def locate_pilot(input_root):
    candidates = []
    for path in Path(input_root).rglob("policy_before_test.json"):
        if path.parent.name != "artifacts" or path.parent.parent.name != HEAD:
            continue
        policy = json.loads(path.read_text())
        if policy.get("base_digest") == EXPECTED["base_digest"]:
            candidates.append(path.parents[3])
    if len(candidates) != 1:
        raise ValueError(
            "Attach exactly the complete CIFAR100N temporal residual pilot seed42 Output; summary-only input is insufficient"
        )
    pilot = candidates[0]
    policy_path = pilot / "heads" / HEAD / "artifacts/policy_before_test.json"
    policy = json.loads(policy_path.read_text())
    if digest(policy_path) != EXPECTED["policy_sha256"]:
        raise ValueError("Input frozen policy bytes differ from the reviewed pilot")
    for key in ("base_digest", "checkpoint_sha256", "cache_manifest_sha256"):
        if policy[key] != EXPECTED[key]:
            raise ValueError(f"Input frozen identity mismatch: {key}")
    if (
        policy["policy"]["alpha"] != 1.0
        or policy["teacher_policy"]["alpha"] != 1.0
        or policy["test_accessed"] is not False
    ):
        raise ValueError("Only the already selected alpha1 policy may be evaluated")
    required = [
        pilot / "artifacts/source_snapshot.zip",
        pilot / "artifacts/source_manifest.json",
        pilot / "artifacts/kernel_witness.json",
        pilot / "evidence/manifest.json",
        pilot / "bases" / BASE / "checkpoints/best.pt",
        pilot / "heads" / HEAD / "checkpoints/best.pt",
    ]
    for path in required:
        if not path.is_file():
            raise ValueError(f"Missing full Output artifact: {path}")
    if digest(required[0]) != EXPECTED["source_archive_sha256"]:
        raise ValueError("Input source archive differs from the trained pilot")
    if digest(pilot / "evidence/manifest.json") != EXPECTED["cache_manifest_sha256"]:
        raise ValueError("Input evidence manifest differs from the frozen policy")
    return pilot


def restore_source(pilot, directory):
    directory = Path(directory)
    if directory.exists():
        raise ValueError("Use a fresh evaluation source directory")
    manifest = json.loads((pilot / "artifacts/source_manifest.json").read_text())
    if manifest["source_archive_sha256"] != EXPECTED["source_archive_sha256"]:
        raise ValueError("Source manifest does not describe the trained code")
    with zipfile.ZipFile(pilot / "artifacts/source_snapshot.zip") as archive:
        if sorted(archive.namelist()) != sorted(manifest["files"]):
            raise ValueError("Source archive membership differs")
        for name, expected in manifest["files"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe source archive name")
            value = archive.read(name)
            if hashlib.sha256(value).hexdigest() != expected:
                raise ValueError("Source file SHA256 differs")
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    return directory


def finish_exports(out, pilot):
    import numpy as np
    from room_for_doubt.evaluation.selection import paired_accuracy_interval

    run = out / "heads" / HEAD / "final_evaluation"
    metrics = json.loads((run / "metrics.json").read_text())
    with np.load(run / "predictions.npz", allow_pickle=False) as data:
        labels = data["labels"]
        if labels.shape != (10000,) or data["base"].shape != (10000, 100):
            raise ValueError("Require the complete official CIFAR100 test set")
        pairs = []
        for baseline, proposed in (
            ("base", "corrected"),
            ("base", "snapshot_mean"),
            ("snapshot_mean", "corrected"),
        ):
            result = paired_accuracy_interval(
                labels, data[baseline].argmax(1), data[proposed].argmax(1)
            )
            pairs.append(dict(baseline=baseline, proposed=proposed, **result))
    metrics["paired_comparisons"] = pairs
    metrics["test_samples"] = 10000
    metrics["paired_scope"] = (
        "2000 nominal paired-example bootstrap draws; conditional on one fitted base/head; no multiplicity adjustment"
    )
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (run / "tables/paired_comparisons.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(pairs)
    shutil.copyfile(
        pilot / "heads" / HEAD / "artifacts/policy_before_test.json",
        out / "policy_before_test.json",
    )
    config = dict(
        run_id=out.name,
        dataset=dict(name="cifar100n", num_classes=100),
        stage="evaluate_only",
        training_performed=False,
        input_pilot=str(pilot),
        expected=EXPECTED,
        fixed_alpha=1.0,
        teacher_epochs=[1, 2, 3, 4, 5],
        comparisons_predeclared=[
            "student_vs_base",
            "teacher_vs_base",
            "student_vs_teacher",
        ],
    )
    from omegaconf import OmegaConf

    OmegaConf.save(OmegaConf.create(config), out / "config.yaml")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "run_metadata.json").write_text(
        json.dumps(
            dict(
                expected=EXPECTED,
                input_pilot=str(pilot),
                policy_sha256=digest(out / "policy_before_test.json"),
                prediction_sha256=digest(run / "predictions.npz"),
                training_performed=False,
                policy_changed=False,
            ),
            indent=2,
        )
    )
    shutil.copyfile(run / "report.md", out / "report.md")
    with (out / "report.md").open("a") as handle:
        handle.write(
            "\n## Predeclared paired comparisons\n\nThese nominal paired-example intervals condition on this one fitted base/head; no multiplicity adjustment. Raw predictions are included in the full results ZIP. No hyperparameter was selected on test.\n"
        )
    lines = [
        "Status: complete",
        "Training: none",
        "Frozen alpha:1",
        "Test samples:10000",
    ]
    for name, result in metrics["methods"].items():
        lines.append(
            f"{name}: accuracy={100 * result['acc']:.4g}% NLL={result['nll']:.4g}"
        )
    for pair in pairs:
        lines.append(
            f"{pair['proposed']} minus {pair['baseline']}: {pair['gain_pp']:+.4g}pp [{pair['ci_low_pp']:+.4g},{pair['ci_high_pp']:+.4g}]"
        )
    (out / "morning_summary.txt").write_text("\n".join(lines) + "\n")
    for full in (False, True):
        filename = "temporal_test_results.zip" if full else "temporal_test_summary.zip"
        with zipfile.ZipFile(out / filename, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                if path.is_file() and (
                    path.suffix in {".json", ".yaml", ".csv", ".md", ".txt", ".log"}
                    or (full and path.name == "predictions.npz")
                ):
                    archive.write(path, str(path.relative_to(out)))
    print((out / "morning_summary.txt").read_text(), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output", default="/kaggle/working/rfd_temporal_test_seed42")
    parser.add_argument(
        "--source-dir", default="/kaggle/working/rfd_frozen_temporal_source"
    )
    parser.add_argument("--data-dir", default="/kaggle/temp/rfd_temporal_test_data")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA accelerator before evaluation")
    pilot = locate_pilot(args.input_root)
    witness = json.loads((pilot / "artifacts/kernel_witness.json").read_text())
    if torch.__version__ != witness["torch"]:
        raise RuntimeError(
            "Torch version differs from pilot; review environment before testing"
        )
    source = restore_source(pilot, args.source_dir)
    out = Path(args.output)
    if out.exists():
        raise ValueError("Preserve the prior evaluation Output; use a fresh session")
    out.mkdir(parents=True)
    shutil.copytree(pilot / "heads" / HEAD, out / "heads" / HEAD)
    env = dict(
        os.environ,
        PYTHONPATH=str(source / "src"),
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(out / "matplotlib"),
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
    )
    print(
        "Frozen pilot verified. Starting only clean-test evaluation, no training.",
        flush=True,
    )
    with (out / "evaluation.log").open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                str(source / "scripts/run.py"),
                "distill-temporal",
                "stage=evaluate",
                f"source_run={pilot / 'bases' / BASE}",
                f"cache_dir={pilot / 'evidence'}",
                f"output_dir={out / 'heads'}",
                f"data_dir={args.data_dir}",
                "device=cuda",
            ],
            cwd=source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(
            "Evaluation failed; preserved evaluation.log and frozen artifacts"
        )
    sys.path.insert(0, str(source / "src"))
    finish_exports(out, pilot)


if __name__ == "__main__":
    main()
