"""Two independent jobs: existing C10 base42 controls or the missing base44."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from omegaconf import OmegaConf

import kaggle_followups as followups


ROOT = Path(__file__).resolve().parents[1]


def complete(path):
    path = Path(path) / "metrics.json"
    return path.exists() and json.loads(path.read_text()).get("status") == "complete"


def matching_bases(input_root, expected_digest):
    """Only a completed exact base42 can stand in for a missing historical study."""
    found = []
    for path in sorted(Path(input_root).rglob("config.yaml")):
        if "heads" in path.parts or ".hydra" in path.parts:
            continue
        cfg = OmegaConf.load(path)
        if (not OmegaConf.is_dict(cfg) or not OmegaConf.is_dict(cfg.get("model"))
                or not OmegaConf.is_dict(cfg.get("dataset"))
                or cfg.model.get("name") != "resnet50"
                or cfg.dataset.get("name") != "cifar10n"
                or cfg.dataset.get("noisy_split") != "worse"
                or cfg.get("seed") != 42 or cfg.get("epochs") != 50
                or not complete(path.parent)):
            continue
        metrics = json.loads((path.parent / "metrics.json").read_text())
        required = ["checkpoints/best.pt", "history.csv", "artifacts/meta_targets.npz",
                    "artifacts/split_ids.npz"]
        if (metrics.get("completed_epochs") == 50
                and all((path.parent / p).exists() for p in required)
                and followups.digest(path.parent / "checkpoints/best.pt") == expected_digest):
            found.append(str(path.parent))
    return found


def inspect_control42(input_root, expected_digest):
    report = followups.inspect_inputs(input_root, dict(cifar10n_seed42=expected_digest))
    studies = [row for row in report["studies"] if row["valid"]]
    bases = matching_bases(input_root, expected_digest) if not studies else []
    return dict(ready_studies=studies, matching_completed_bases=bases,
                can_start=len(studies) == 1 or (not studies and bool(bases)),
                rebuilding_heads=not studies and bool(bases),
                note="No notebook Output download and no base42 retraining.")


def link_data(inputs, destination):
    """Reuse mounted images/labels; fresh seed44 may download official data normally."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("cifar-10-batches-py", "CIFAR-10_human.pt"):
        target = destination / name
        if target.is_symlink() and not target.exists():
            target.unlink()
        if target.exists():
            continue
        matches = sorted(Path(inputs).rglob(name))
        if matches:
            target.symlink_to(matches[0].resolve(), target_is_directory=matches[0].is_dir())


def base_config(settings, out, data):
    cfg = OmegaConf.load(ROOT / "configs/kaggle_base.yaml")
    cfg.run_id, cfg.output_dir, cfg.data_dir = "cifar10n_seed44", str(out), str(data)
    cfg.dataset = dict(name="cifar10n", noisy_split="worse", num_classes=10, val_ratio=.1)
    cfg.seed, cfg.base_split_seed = settings.base_seed, settings.base_split_seed
    cfg.max_hours = settings.base_max_hours
    return cfg


def study_config(base, out, data, hours):
    cfg = OmegaConf.load(ROOT / "configs/feature_study.yaml")
    cfg.source_run, cfg.post_run = str(base), None
    cfg.run_id, cfg.output_dir, cfg.data_dir = out.name, str(out.parent), str(data)
    cfg.device, cfg.max_hours = "cuda", hours
    return cfg


def reconstruction_table(study, destination):
    """Record differences, never choose an old/new reconstruction by test accuracy."""
    old = ROOT / "paper/evidence/c10_s42/test_validation_selected.csv"
    new = Path(study) / "tables/test_validation_selected.csv"
    with old.open() as stream:
        reference = {r["method"]: r for r in csv.DictReader(stream)}
    with new.open() as stream:
        rows = list(csv.DictReader(stream))
    comparison = []
    for row in rows:
        if row["method"] in reference:
            previous = reference[row["method"]]
            comparison.append(dict(
                method=row["method"], original_acc=previous["acc"], rebuilt_acc=row["acc"],
                original_views=previous["views"], rebuilt_views=row["views"],
                original_temperature=previous["temperature"], rebuilt_temperature=row["temperature"],
                gain_pp=100 * (float(row["acc"]) - float(previous["acc"])),
            ))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["control42", "seed44"], required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect-inputs", action="store_true")
    args = parser.parse_args()
    settings = OmegaConf.load(ROOT / "configs/cifar10_closeout.yaml")
    inspection = inspect_control42(args.input_root, settings.control42_digest) if args.task == "control42" else dict(can_start=True, new_base_seed=44)
    if args.inspect_inputs:
        print(json.dumps(inspection, indent=2))
        return
    out = (args.output or Path("/kaggle/working/rfd_cifar10_" + args.task)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    contract = dict(task=args.task, settings=OmegaConf.to_container(settings),
                    git_commit=commit, input_root=str(args.input_root), output=str(out))
    saved = out / "config.yaml"
    if saved.exists() and OmegaConf.to_container(OmegaConf.load(saved)) != contract:
        raise ValueError("Existing closeout run configuration differs")
    OmegaConf.save(OmegaConf.create(contract), saved, resolve=True)
    (out / "input_inspection.json").write_text(json.dumps(inspection, indent=2))
    state = dict(status="running", task=args.task, stages=[], inspection=inspection)
    started = time.monotonic()

    def export():
        state["elapsed_seconds"] = time.monotonic() - started
        (out / "metrics.json").write_text(json.dumps(state, indent=2))
        (out / "run_metadata.json").write_text(json.dumps(contract, indent=2))
        (out / "morning_summary.txt").write_text(
            "Status: " + state["status"] + "\n" + "\n".join(
                f"{r['stage']}: {r['status']}" for r in state["stages"]) + "\n")
        (out / "report.md").write_text(
            "# Experiment Report: cifar10_" + args.task
            + "\n\n## Configuration\nSee config.yaml.\n\n## Dataset\nCIFAR-10N Worse; fixed base split42."
              "\n\n## Checkpoint\nMinimum noisy-validation cross entropy."
              "\n\n## Metrics\n" + json.dumps(state, indent=2)
            + "\n\n## Generated Artifacts\nFull stage Outputs, compact tables and summary ZIP."
              "\n\n## Notes\nIf base42 heads are rebuilt, these are new fits on the same base,"
              " not recovered original weights or independent base seeds. Their original/rebuilt"
              " comparison is descriptive; it never selects which fitted heads to report.\n")
        with zipfile.ZipFile(out / "closeout_summary.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                if (path.is_file() and path.suffix in {".yaml", ".json", ".csv", ".md", ".txt"}
                        and ".hydra" not in path.parts):
                    archive.write(path, str(path.relative_to(out)))

    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
               MPLCONFIGDIR="/kaggle/working/matplotlib", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")

    def stage(name, cfg, target, command=None):
        if complete(target):
            state["stages"].append(dict(stage=name, status="complete_reused"))
            export()
            return
        if settings.max_hours - (time.monotonic() - started) / 3600 < .25:
            raise TimeoutError("Session admission budget exhausted; preserve full Output")
        launch = out / "launches" / (name + ".yaml")
        launch.parent.mkdir(exist_ok=True)
        OmegaConf.save(cfg, launch, resolve=True)
        if command is None:
            cmd = [sys.executable, "-m", "room_for_doubt.analysis.campaign_stage", name, str(launch), str(target)]
        else:
            cmd = [sys.executable, str(ROOT / "scripts/run.py"), command,
                   "--config-path=" + str(launch.parent), "--config-name=" + launch.stem]
        print("START", args.task, name, flush=True)
        begin = time.monotonic()
        with (out / (name + ".log")).open("w") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        done = code == 0 and complete(target)
        state["stages"].append(dict(stage=name, status="complete" if done else "incomplete",
                                     seconds=time.monotonic()-begin, returncode=code))
        export()
        if not done:
            raise RuntimeError("Stage incomplete: " + name + "; preserve full Output before resuming")

    export()
    try:
        if not inspection["can_start"]:
            state["status"] = "blocked_missing_verified_base42"
            export()
            print("Не найдено готовое исследование или завершённая база CIFAR-10N seed42 с нужной SHA256.", flush=True)
            print("Обучение не начато. Нужен полный Output завершённой базы42, не summary.", flush=True)
            return
        import torch
        if not torch.cuda.is_available() or (args.task == "seed44" and torch.cuda.device_count() != 2):
            raise RuntimeError("Enable GPU; choose T4 x2 for the matched seed44 base training")
        if shutil.disk_usage(out).free < settings.min_free_gb * 1024**3:
            raise RuntimeError("Insufficient local disk")
        data = out / "data"
        link_data(args.input_root, data)
        if args.task == "seed44":
            base = out / "base"
            stage("base", base_config(settings, base, data), base)
            study, post = out / "feature_study", None
        elif inspection["ready_studies"]:
            row = inspection["ready_studies"][0]
            base, study, post = Path(row["base"]), Path(row["study"]), Path(row["post"])
        else:
            base = Path(inspection["matching_completed_bases"][0])
            study, post = out / "rebuilt_feature_study", None
        from room_for_doubt.data.cifar_noisy import load_observed_labels
        from room_for_doubt.data.kaggle_artifacts import load_completed_run
        noisy = load_observed_labels(data, "cifar10n", "worse")
        bundle = load_completed_run(base, noisy)
        if args.task == "control42" and bundle["checkpoint_digest"] != settings.control42_digest:
            raise ValueError("Base42 checkpoint identity differs")
        state["base_digest"] = bundle["checkpoint_digest"]
        state["heads_rebuilt"] = args.task == "control42" and not inspection["ready_studies"]
        del bundle, noisy
        if post is None:
            hours = min(settings.study_max_hours, settings.max_hours-(time.monotonic()-started)/3600-.1)
            stage("study", study_config(base, study, data, hours), study)
            post = study / "shared"
        if state["heads_rebuilt"]:
            reconstruction_table(study, out / "tables/original_vs_rebuilt.csv")
        paired = out / "paired"
        cfg = OmegaConf.load(ROOT / "configs/feature_contrasts.yaml")
        cfg.run_id, cfg.output_dir, cfg.study_run, cfg.post_run = "paired", str(out), str(study), str(post)
        stage("paired", cfg, paired, "contrast-feature-targets")
        control = out / "augmentation"
        cfg = OmegaConf.load(ROOT / "configs/augmentation_control.yaml")
        cfg.run_id, cfg.output_dir = "augmentation", str(out)
        cfg.source_run, cfg.study_run, cfg.post_run = str(base), str(study), str(post)
        cfg.data_dir, cfg.device = str(data), "cuda"
        cfg.max_hours = min(settings.control_max_hours, settings.max_hours-(time.monotonic()-started)/3600-.05)
        stage("augmentation", cfg, control, "study-augmentation-control")
        state["status"] = "complete"
    except Exception as exc:
        state["status"], state["error"] = "incomplete", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        export()
        print("Сводка:", out / "closeout_summary.zip", flush=True)


if __name__ == "__main__":
    main()
