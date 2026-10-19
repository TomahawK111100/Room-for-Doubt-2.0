"""Bounded unattended base-training and matched downstream jobs in one notebook."""

import json
import shutil
import time
import zipfile
import subprocess
import sys
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from room_for_doubt.training.kaggle_base import resume_contract
from room_for_doubt.evaluation.campaign_ensemble import evaluate_ensembles
from room_for_doubt.reporting.artifacts import (
    save_metrics,
    save_run_metadata,
    generate_report,
    write_csv,
)
from room_for_doubt.utils.paths import config_directory


def isolated_stage(cfg, out, stage):
    """Release all stage-owned CPU/GPU allocations when its process exits."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = out / "stage_launch.yaml"
    OmegaConf.save(cfg, config, resolve=True)
    print(f"Starting isolated {stage}: {out}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "room_for_doubt.analysis.campaign_stage",
         stage, str(config), str(out)], check=True,
    )
    return json.loads((out / "metrics.json").read_text())


def train_base(cfg, out):
    return isolated_stage(cfg, out, "base")


def run_study(cfg, out):
    return isolated_stage(cfg, out, "study")


def run_path(out, job):
    return Path(out) / "jobs" / f"{job['dataset']}_seed{job['seed']}"


def export_campaign(out, cfg, state):
    out = Path(out)
    config = OmegaConf.to_container(cfg, resolve=True)
    save_metrics(out, state)
    save_run_metadata(
        out, cfg.run_id, config, checkpoint_path="jobs/*/base/checkpoints/best.pt"
    )
    generate_report(
        out,
        cfg.run_id,
        config,
        state,
        notes=(
            "Fixed job order recorded in config.yaml. "
            "All runs retain training recipes and fixed train/validation split seed42. "
            "Changing model initialization does not change held-out membership. "
            "Each completed base is followed by the fixed standard 2x3x3 feature study. "
            "Budget pauses/deferred jobs are reported explicitly; completion does not imply optional jobs ran. "
            "No model is selected by test performance. Ensemble membership/order is predeclared; "
            "ensemble intervals are conditional and multi-model inference has greater compute cost. "
            "A campaign still needs both notebooks' bundles for the cross-base comparison including the original seed42 feature heads."
        ),
    )
    compact = []
    for job in state["jobs"]:
        compact.append(
            {
                k: job.get(k)
                for k in (
                    "dataset",
                    "seed",
                    "optional",
                    "status",
                    "elapsed_seconds",
                    "base_test_acc",
                    "study_run",
                )
            }
        )
    if compact:
        write_csv(out / "tables/job_status.csv", compact)
    results = []
    for job in state["jobs"]:
        if job["status"] == "complete":
            import csv

            with (Path(job["study_run"]) / "tables/aggregate.csv").open() as handle:
                results.extend(
                    dict(dataset=job["dataset"], base_seed=job["seed"], **row)
                    for row in csv.DictReader(handle)
                )
    if results:
        write_csv(out / "tables/downstream_comparison.csv", results)
    lines = [f"Status: {state['status']}"]
    for job in state["jobs"]:
        lines.append(
            f"{job['dataset']} seed{job['seed']}: {job['status']}, base_acc={job.get('base_test_acc')}"
        )
    (out / "morning_summary.txt").write_text("\n".join(lines) + "\n")
    with zipfile.ZipFile(
        out / "campaign_summary.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in (
            "config.yaml",
            "metrics.json",
            "run_metadata.json",
            "report.md",
            "morning_summary.txt",
        ):
            path = out / name
            if path.exists():
                archive.write(path, name)
        for path in (out / "tables").glob("*.csv"):
            archive.write(path, str(path.relative_to(out)))
        for path in (out / "jobs").glob("*/feature_study/summary_bundle.zip"):
            archive.write(path, str(path.relative_to(out)))
        recipe = out / "artifacts/source_config.yaml"
        if recipe.exists():
            archive.write(recipe, str(recipe.relative_to(out)))
        for base in (out / "jobs").glob("*/base"):
            for name in (
                "config.yaml",
                "metrics.json",
                "run_metadata.json",
                "history.csv",
                "report.md",
            ):
                path = base / name
                if path.exists():
                    archive.write(path, str(path.relative_to(out)))
    print(
        "\n" + "\n".join(lines) + f"\nDownload: {out / 'campaign_summary.zip'}",
        flush=True,
    )


def run_campaign(cfg, out):
    started = time.monotonic()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("tables", "artifacts", "jobs"):
        (out / name).mkdir(exist_ok=True)
    if not cfg.source_run or not cfg.post_run or cfg.max_hours <= 0:
        raise ValueError("Source base/post and positive campaign budget required")
    planned = OmegaConf.to_container(cfg.jobs, resolve=True)
    expected = {
        "independent_runs": [("cifar100n", 43), ("cifar10n", 42), ("cifar100n", 44)],
        "replication_cifar100": [("cifar100n", 44)],
        "replication_cifar10": [("cifar10n", 43), ("cifar10n", 44)],
    }.get(cfg.run_id)
    if [(job["dataset"], job["seed"]) for job in planned] != expected:
        raise ValueError("Use fixed campaign job order")
    cfg.dataset = {"name": "cifar100n+cifar10n"}
    config = OmegaConf.to_container(cfg, resolve=True)
    config_path = out / "config.yaml"
    if config_path.exists():
        prior = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        for key in ("jobs", "base_split_seed"):
            if prior[key] != config[key]:
                raise ValueError("Existing campaign scientific config differs")
    OmegaConf.save(cfg, config_path, resolve=True)
    source_config = OmegaConf.load(Path(cfg.source_run) / "config.yaml")
    if (
        source_config.epochs != 50
        or source_config.dataset.name != "cifar100n"
        or source_config.seed != 42
    ):
        raise ValueError("Campaign requires the recorded 50-epoch training recipe")
    recipe = out / "artifacts/source_config.yaml"
    if recipe.exists() and resume_contract(
        OmegaConf.to_container(OmegaConf.load(recipe), resolve=True)
    ) != resume_contract(OmegaConf.to_container(source_config, resolve=True)):
        raise ValueError("Campaign source training recipe differs from prior run")
    OmegaConf.save(source_config, recipe, resolve=True)
    data_dir = Path(cfg.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    state = dict(
        status="running", jobs=[dict(**job, status="pending") for job in planned]
    )
    recent_durations = []

    def remaining():
        return cfg.max_hours - (time.monotonic() - started) / 3600

    for job in state["jobs"]:
        root = run_path(out, job)
        base, study = root / "base", root / "feature_study"
        job["study_run"] = str(study)
        saved_study = study / "metrics.json"
        if (
            saved_study.exists()
            and json.loads(saved_study.read_text()).get("status") == "complete"
        ):
            job["status"] = "complete"
            job["base_test_acc"] = json.loads((base / "metrics.json").read_text())[
                "test_acc"
            ]
            continue
        required_hours = max(
            cfg.minimum_start_hours, 1.15 * max(recent_durations, default=0)
        )
        base_complete = (base / "metrics.json").exists() and json.loads(
            (base / "metrics.json").read_text()
        ).get("status") == "complete"
        threshold = 0.5 if base_complete else required_hours
        if remaining() < threshold:
            job["status"] = "deferred_time_budget"
            continue
        if shutil.disk_usage(out).free < cfg.min_free_gb * 1024**3:
            job["status"] = "deferred_disk_budget"
            continue
        tick = time.monotonic()
        job["status"] = "running"
        export_campaign(out, cfg, state)
        try:
            if not base_complete:
                local = OmegaConf.create(
                    OmegaConf.to_container(source_config, resolve=True)
                )
                local.seed, local.base_split_seed = job["seed"], cfg.base_split_seed
                local.run_id, local.data_dir, local.output_dir = (
                    root.name,
                    str(data_dir),
                    str(root),
                )
                local.dataset = dict(
                    name=job["dataset"],
                    noisy_split="noisy100"
                    if job["dataset"] == "cifar100n"
                    else "worse",
                    num_classes=100 if job["dataset"] == "cifar100n" else 10,
                    val_ratio=0.1,
                )
                local.resume, local.require_cuda = "auto", True
                local.max_hours = max(0.01, min(cfg.base_max_hours, remaining() - 0.1))
                result = train_base(local, base)
                if result["status"] != "complete":
                    job["status"] = "paused_base_training"
                    break
            base_metrics = json.loads((base / "metrics.json").read_text())
            job["base_test_acc"] = base_metrics["test_acc"]
            if remaining() < 0.25:
                job["status"] = "paused_before_downstream"
                break
            downstream = OmegaConf.load(config_directory() / "feature_study.yaml")
            downstream.source_run, downstream.post_run = str(base), None
            downstream.run_id, downstream.output_dir = (
                root.name + "_features",
                str(root),
            )
            downstream.data_dir, downstream.device = str(data_dir), "cuda"
            downstream.max_hours = max(0.01, remaining() - 0.05)
            result = run_study(downstream, study)
            job["status"] = (
                "complete" if result["status"] == "complete" else "paused_downstream"
            )
            job["elapsed_seconds"] = time.monotonic() - tick
            recent_durations.append(job["elapsed_seconds"] / 3600)
            export_campaign(out, cfg, state)
            if job["status"] != "complete":
                break
        except Exception as exc:
            job["status"], job["error"] = "failed", f"{type(exc).__name__}: {exc}"
            job["elapsed_seconds"] = time.monotonic() - tick
            # Independent jobs may continue; the failure is preserved, not hidden.
            export_campaign(out, cfg, state)
            if isinstance(exc, subprocess.CalledProcessError) and exc.returncode < 0:
                # A killed worker is a resource failure; preserve progress and
                # do not spend the remaining quota starting more GPU jobs.
                break
    required_complete = all(
        job["status"] == "complete" for job in state["jobs"] if not job["optional"]
    )
    all_complete = all(job["status"] == "complete" for job in state["jobs"])
    state["status"] = (
        "complete"
        if all_complete
        else ("required_complete_optional_deferred" if required_complete else "partial")
    )
    state["elapsed_seconds"] = time.monotonic() - started
    completed = [job for job in state["jobs"] if job["status"] == "complete"]
    # This is a fixed baseline, never a test-based selection of the best seeds.
    state["ensemble_results"] = evaluate_ensembles(cfg.post_run, completed, out)
    export_campaign(out, cfg, state)
    return state


@hydra.main(
    config_path=str(config_directory()),
    config_name="kaggle_campaign",
    version_base=None,
)
def main(cfg):
    run_campaign(cfg, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
