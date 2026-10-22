"""One frozen Animal-10N pilot: audit inputs, base, matched heads and controls."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/animal10n_pilot.yaml"


def locate_data(root):
    """Accept one binary release only; never guess between multiple copies."""
    root = Path(root).resolve()
    candidates = {
        p.parent
        for p in root.rglob("test_batch.bin")
        if list(p.parent.glob("data_batch_*.bin"))
    }
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one Animal-10N binary release under {root}; found {len(candidates)}"
        )
    return candidates.pop()


def complete(path):
    path = Path(path) / "metrics.json"
    return path.exists() and json.loads(path.read_text()).get("status") == "complete"


def test_evaluation_seen(out):
    """Record persisted test use even if a later stage fails or is paused."""
    out = Path(out)
    for stage in ("base", "feature_study", "paired", "augmentation"):
        p = out / stage / "metrics.json"
        if p.exists():
            metrics = json.loads(p.read_text())
            if (
                metrics.get("test_evaluated")
                or metrics.get("test_acc") is not None
                or (stage == "paired" and metrics.get("status") == "complete")
            ):
                return True
    return (out / "base/artifacts/test_predictions.npz").exists()


def cuda_sanity(out):
    """Check the declared two-GPU AMP/backward path, without dataset evaluation."""
    import torch

    for module in (
        "training.kaggle_base",
        "analysis.feature_study",
        "analysis.feature_contrasts",
        "analysis.augmentation_control",
    ):
        importlib.import_module("room_for_doubt." + module)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("This frozen base recipe requires two visible CUDA GPUs")
    model = torch.nn.DataParallel(
        torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)
        ).cuda()
    )
    x = torch.randn(8, 8, device="cuda")
    with torch.autocast("cuda"):
        value = model(x).float().square().mean()
    value.backward()
    finite = torch.isfinite(value).item() and all(
        p.grad is not None and torch.isfinite(p.grad).all().item()
        for p in model.parameters()
    )
    if not finite:
        raise FloatingPointError(
            "Two-GPU AMP sanity produced nonfinite values/gradients"
        )
    environment = dict(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        nvidia_driver_versions=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=15,
        )
        .strip()
        .splitlines(),
        gpu_names=[torch.cuda.get_device_name(i) for i in range(2)],
        packages={
            p: importlib.metadata.version(p)
            for p in (
                "torchvision",
                "hydra-core",
                "numpy",
                "scikit-learn",
                "pandas",
                "matplotlib",
            )
        },
        sanity="two_gpu_amp_backward_passed",
        evaluation_type="simulation_only",
        full_training_memory_validated=False,
    )
    (Path(out) / "artifacts/environment.json").write_text(
        json.dumps(environment, indent=2) + "\n"
    )
    del model, x, value
    torch.cuda.empty_cache()


def make_config(settings, name, out, data, seed, manifest_sha256):
    cfg = OmegaConf.create(OmegaConf.to_container(settings[name], resolve=True))
    if name == "base":
        cfg.run_id, cfg.output_dir = f"animal10n_seed{seed}", str(out / "base")
        cfg.data_dir, cfg.seed = str(data), seed
        cfg.dataset.manifest_sha256 = manifest_sha256
    else:
        cfg.output_dir = str(out)
        cfg.run_id = "feature_study" if name == "study" else name
        if name in ("study", "augmentation"):
            cfg.source_run, cfg.data_dir = str(out / "base"), str(data)
        if name in ("paired", "augmentation"):
            cfg.study_run = str(out / "feature_study")
            cfg.post_run = str(out / "feature_study/shared")
        if name in ("paired", "augmentation"):
            cfg.hydra = dict(run=dict(dir=str(out / name)))
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument(
        "--output", type=Path, default=Path("/kaggle/working/rfd_animal10n_seed42")
    )
    parser.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()
    from room_for_doubt.data.animal10n import Animal10N, file_manifest
    from room_for_doubt.reporting.artifacts import (
        generate_report,
        save_run_metadata,
        save_metrics,
    )

    settings = OmegaConf.load(PROTOCOL)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)
    started = time.monotonic()
    state = dict(status="inspecting", seed=args.seed, stages=[], test_evaluated=False)
    config = dict(
        protocol_version=settings.protocol_version,
        seed=args.seed,
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        requested_scope="base_only"
        if args.base_only
        else "base_study_paired_augmentation",
        settings=OmegaConf.to_container(settings, resolve=True),
    )
    previous = out / "config.yaml"
    if previous.exists() and OmegaConf.to_container(OmegaConf.load(previous)) != config:
        raise ValueError(
            "Existing pilot uses a different frozen protocol or requested scope"
        )
    OmegaConf.save(OmegaConf.create(config), previous, resolve=True)
    shutil.copy2(PROTOCOL, out / "artifacts/frozen_protocol.yaml")

    def export():
        state["test_evaluated"] = state["test_evaluated"] or test_evaluation_seen(out)
        state["elapsed_seconds"] = time.monotonic() - started
        save_metrics(out, state)
        save_run_metadata(
            out,
            out.name,
            config,
            device="not initialized" if args.inspect_only else "cuda",
            input_root=str(args.input_root),
            source_protocol_sha256=config["protocol_sha256"],
        )
        generate_report(
            out,
            out.name,
            config,
            state,
            {},
            "Input audit does not fit a model or select a policy. Stage completion requires "
            "saved outputs. Test metrics, if produced by a completed stage, do not select "
            "targets, view grids, hyperparameters or further dataset choices. Animal-10N "
            "provides no full clean-training-label vector; train-noise detection is unavailable.",
        )
        (out / "morning_summary.txt").write_text(
            "Status: "
            + state["status"]
            + "\n"
            + "\n".join(f"{r['stage']}: {r['status']}" for r in state["stages"])
            + ("\n" + state["error"] if "error" in state else "")
            + "\n"
        )
        with zipfile.ZipFile(
            out / "animal10n_summary.zip", "w", zipfile.ZIP_DEFLATED
        ) as z:
            for p in sorted(out.rglob("*")):
                if (
                    p.is_file()
                    and p.suffix in (".csv", ".json", ".yaml", ".md", ".txt")
                    and "data" not in p.relative_to(out).parts
                    and ".hydra" not in p.parts
                    and "heads" not in p.parts
                ):
                    z.write(p, str(p.relative_to(out)))

    def stage(name, cfg, target, command=None):
        if complete(target):
            state["stages"].append(dict(stage=name, status="complete_reused"))
            return
        remaining = settings.max_hours - (time.monotonic() - started) / 3600
        if remaining < 0.25:
            raise TimeoutError(
                "Session admission limit reached; preserve full Output to resume"
            )
        if "max_hours" in cfg:
            cfg.max_hours = min(cfg.max_hours, remaining - 0.05)
        launch = out / "launches" / (name + ".yaml")
        launch.parent.mkdir(exist_ok=True)
        OmegaConf.save(cfg, launch, resolve=True)
        cmd = (
            [
                sys.executable,
                "-m",
                "room_for_doubt.analysis.campaign_stage",
                name,
                str(launch),
                str(target),
            ]
            if command is None
            else [
                sys.executable,
                str(ROOT / "scripts/run.py"),
                command,
                "--config-path=" + str(launch.parent),
                "--config-name=" + launch.stem,
            ]
        )
        env = dict(
            os.environ,
            PYTHONUNBUFFERED="1",
            MPLBACKEND="Agg",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="4",
            MPLCONFIGDIR=str(out / "matplotlib"),
        )
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        print("START", name, flush=True)
        tick = time.monotonic()
        with (out / (name + ".log")).open("w") as log:
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = process.wait()
        done = code == 0 and complete(target)
        state["stages"].append(
            dict(
                stage=name,
                status="complete" if done else "incomplete",
                seconds=time.monotonic() - tick,
                returncode=code,
            )
        )
        export()
        if not done:
            raise RuntimeError("Incomplete stage " + name + "; preserve full Output")

    try:
        data = locate_data(args.input_root)
        manifest = file_manifest(data)
        # Read only observed training labels before any fitting. Test is opaque bytes here.
        raw = Animal10N(data, train=True)
        manifest["observed_train_class_counts"] = [
            int((raw.targets == c).sum()) for c in range(10)
        ]
        del raw
        (out / "artifacts/input_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        state["data_manifest_sha256"] = manifest["manifest_sha256"]
        state["data_dir"] = str(data)
        if args.inspect_only:
            state["status"] = "ready_to_run_not_trained"
            return
        if shutil.disk_usage(out).free < settings.min_free_gb * 1024**3:
            raise RuntimeError("At least 8 GiB free working disk is required")
        cuda_sanity(out)
        state["cuda_sanity"] = "passed_simulation_only"
        for name, directory, command in [
            ("base", "base", None),
            ("study", "feature_study", None),
            ("paired", "paired", "contrast-feature-targets"),
            ("augmentation", "augmentation", "study-augmentation-control"),
        ]:
            if args.base_only and name != "base":
                break
            cfg = make_config(
                settings, name, out, data, args.seed, manifest["manifest_sha256"]
            )
            stage(name, cfg, out / directory, command)
            if name == "base":
                from room_for_doubt.data.cifar_noisy import load_observed_labels
                from room_for_doubt.data.kaggle_artifacts import load_completed_run

                bundle = load_completed_run(
                    out / "base", load_observed_labels(data, "animal10n", "human")
                )
                if (
                    bundle["config"]["dataset"].get("manifest_sha256")
                    != manifest["manifest_sha256"]
                ):
                    raise ValueError(
                        "Completed base belongs to different Animal-10N input files"
                    )
                state["base_digest"] = bundle["checkpoint_digest"]
                del bundle
        state["status"] = "base_complete" if args.base_only else "complete"
        state["test_evaluated"] = True
    except Exception as exc:
        state["status"] = (
            "blocked_input_or_runtime" if not state["stages"] else "incomplete"
        )
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        export()
        print(
            "Status:",
            state["status"],
            "Summary:",
            out / "animal10n_summary.zip",
            flush=True,
        )


if __name__ == "__main__":
    main()
