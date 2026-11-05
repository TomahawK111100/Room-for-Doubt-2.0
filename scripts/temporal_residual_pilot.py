"""One unattended CIFAR100N pilot, ending on noisy validation (never test)."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
BASE_ID = "temporal_cifar100n_seed42_v1"
HEAD_ID = "temporal_residual_seed42_v1"


def export_summary(out, metrics, source=None):
    """Exclude checkpoints, images and per-sample arrays from the small handoff."""
    out = Path(out)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "run_metadata.json").write_text(
        json.dumps(
            dict(
                run_id=out.name,
                source=source,
                test_evaluated=False,
                stages=["base", "prepare", "fit"],
                final_evaluation_deferred=True,
            ),
            indent=2,
        )
    )
    (out / "report.md").write_text(
        f"# Experiment Report: {out.name}\n\n## Configuration\nSee config.yaml.\n\n"
        "## Dataset\nCIFAR-100N noisy100, base split seed42.\n\n"
        "## Checkpoint\nBase: minimum noisy-validation CE; head: minimum reserved meta-validation KL.\n\n"
        f"## Metrics\n```json\n{json.dumps(metrics, indent=2)}\n```\n\n"
        "## Compact Results\nSee heads/temporal_residual_seed42_v1/tables/.\n\n"
        "## Generated Artifacts\nBase checkpoints, early teachers, sealed evidence, fitted head and frozen policy.\n\n"
        "## Notes\nNo clean-test dataset was instantiated. No benchmark gain or statistical significance is established.\n"
    )
    lines = [f"Status: {metrics['status']}", "Clean test: not evaluated"]
    if "head_metrics" in metrics:
        head = metrics["head_metrics"]
        lines += [
            f"Diagnosis: {head['pilot_diagnosis']}",
            f"Teacher noisy-val gain: {head['teacher_val_gain_pp']:.4g} pp",
            f"Student noisy-val gain: {head['student_val_gain_pp']:.4g} pp",
            f"Selected alpha: {head['selected_alpha']}",
        ]
    (out / "morning_summary.txt").write_text("\n".join(lines) + "\n")
    # Commit the terminal stage before archiving; never ship a stale fit/running marker.
    (out / "progress.json").write_text(
        json.dumps(
            dict(
                stage=(
                    "finished"
                    if metrics["status"] == "validation_complete_test_deferred"
                    else metrics.get("stage")
                ),
                status=metrics["status"],
                test_evaluated=False,
            )
        )
    )
    archive_path = out / "temporal_summary.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if (
                path.is_file()
                and path.suffix in {".yaml", ".json", ".md", ".txt", ".csv", ".log"}
                and "evidence" not in path.relative_to(out).parts
            ):
                archive.write(path, str(path.relative_to(out)))


def execute(out, data, max_base_hours=6.5):
    out, data = Path(out).resolve(), Path(data).resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        dataset=dict(name="cifar100n", noisy_split="noisy100", num_classes=100),
        seed=42,
        base_split_seed=42,
        teacher_epochs=[1, 2, 3, 4, 5],
        base_epochs=50,
        head_epochs=25,
        head_hidden=256,
        head_alphas=[0.0, 0.25, 0.5, 1.0],
        final_test=False,
        base_config="temporal_base",
        head_config="temporal_residual",
        data_dir=str(data),
        max_base_hours=max_base_hours,
    )
    saved_config = out / "config.yaml"
    if (
        saved_config.exists()
        and OmegaConf.to_container(OmegaConf.load(saved_config)) != protocol
    ):
        raise ValueError("Existing pilot scientific/runtime protocol differs")
    OmegaConf.save(OmegaConf.create(protocol), saved_config)
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        PYTHONUNBUFFERED="1",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(out / "matplotlib"),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
    )
    driver = [sys.executable, str(ROOT / "scripts/run.py")]
    source = out / "bases" / BASE_ID
    cache = out / "evidence"
    head = out / "heads" / HEAD_ID
    commands = [
        (
            "base",
            driver
            + [
                "train-kaggle-base",
                "--config-name",
                "temporal_base",
                f"data_dir={data}",
                f"output_dir={out / 'bases'}",
                "num_workers=0",
                f"max_hours={max_base_hours}",
            ],
        ),
        (
            "prepare",
            driver
            + [
                "distill-temporal",
                "stage=prepare",
                f"source_run={source}",
                f"data_dir={data}",
                f"cache_dir={cache}",
                f"output_dir={out / 'heads'}",
            ],
        ),
        (
            "fit",
            driver
            + [
                "distill-temporal",
                "stage=fit",
                f"cache_dir={cache}",
                f"output_dir={out / 'heads'}",
            ],
        ),
    ]
    current = None
    try:
        for current, command in commands:
            if current == "fit" and (head / "metrics.json").exists():
                existing = json.loads((head / "metrics.json").read_text())
                if existing.get("status") == "policy_frozen":
                    from room_for_doubt.training.temporal_residual import load_frozen
                    import torch

                    load_frozen(head, cache, torch.device("cpu"))
                    continue
            (out / "progress.json").write_text(
                json.dumps(dict(stage=current, status="running", test_evaluated=False))
            )
            print(f"START {current}; log: {out / (current + '.log')}", flush=True)
            with (out / f"{current}.log").open("a") as log:
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                )
            if result.returncode:
                raise RuntimeError(
                    f"{current} exited {result.returncode}; inspect {current}.log"
                )
            if current == "base":
                base = json.loads((source / "metrics.json").read_text())
                if base["status"] != "complete":
                    metrics = dict(
                        status="paused_base_time_budget",
                        stage=current,
                        base_metrics=base,
                        test_evaluated=False,
                    )
                    export_summary(out, metrics)
                    return metrics
                if base.get("test_evaluated") is not False:
                    raise ValueError("Pilot base unexpectedly evaluated test")
            print(f"DONE {current}", flush=True)
        metrics = dict(
            status="validation_complete_test_deferred",
            test_evaluated=False,
            head_metrics=json.loads((head / "metrics.json").read_text()),
        )
        source_manifest = out / "artifacts/source_manifest.json"
        provenance = (
            json.loads(source_manifest.read_text())
            if source_manifest.exists()
            else None
        )
        export_summary(out, metrics, provenance)
        print((out / "morning_summary.txt").read_text(), flush=True)
        print("Download:", out / "temporal_summary.zip", flush=True)
        return metrics
    except Exception as error:
        export_summary(
            out,
            dict(
                status="failed", stage=current, error=str(error), test_evaluated=False
            ),
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-base-hours", type=float, default=6.5)
    args = parser.parse_args()
    execute(args.output, args.data_dir, args.max_base_hours)


if __name__ == "__main__":
    main()
