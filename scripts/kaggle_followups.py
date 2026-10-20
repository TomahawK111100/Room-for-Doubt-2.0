"""Run bounded follow-up checks on the two completed replication Outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from omegaconf import OmegaConf

EXPECTED = {
    "cifar100n_seed44": "b340fd214bdad727a32a23950c1dd01b7702b7326d2de916752d95a563dba9a0",
    "cifar10n_seed43": "d04d25aad849a4d31e62ce4231b5ee5babe5ea20866105fa8c0d27bf73fc13e7",
}
NOTEBOOKS = [
    "https://www.kaggle.com/code/tomahawkkz/fork-of-fork-of-fork-of-packet-b-tocsin/output",
    "https://www.kaggle.com/code/tomahawkkz/fork-of-fork-of-packet-b-tocsin/output",
]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def study_requirements(cfg):
    required = [
        "metrics.json", "artifacts/policies_before_test.json",
        "artifacts/training_targets.npz",
        "native_features/train_features.npy",
        "projected/base_resnet50/projection.npz",
        "projected/base_resnet50/train.npy",
        "projected/base_resnet50/val.npy",
        "projected/base_resnet50/test.npy",
        "shared/cache/test_complete.json",
        "shared/cache/val_complete.json",
    ]
    for split in ("val", "test"):
        required += [
            f"native_features/{split}_complete.json",
            *[f"shared/cache/{split}_{suffix}.npy" for suffix in ("probs", "ids", "labels")],
        ]
    for encoder in cfg.encoders:
        for mode in cfg.target_modes:
            for seed in cfg.seeds:
                head = f"heads/{encoder}_{mode}_{seed}"
                required += [f"{head}/checkpoints/best.pt", f"{head}/metrics.json",
                             f"{head}/history.csv",
                             f"artifacts/test_{encoder}__{mode}__{seed}.npy"]
    return required

def candidate_studies(root):
    """Use config enumeration for both inspection and execution, including mount layouts."""
    for path in sorted(Path(root).rglob("config.yaml")):
        if "heads" in path.parts or ".hydra" in path.parts:
            continue
        cfg = OmegaConf.load(path)
        if (not OmegaConf.is_dict(cfg) or cfg.get("base_seed") is None
                or not cfg.get("encoders") or not cfg.get("target_modes")
                or not cfg.get("base_digest")):
            continue
        yield path.parent, cfg


def inspect_inputs(root):
    """Run the same full preflight used by execution, without a GPU or training."""
    root = Path(root)
    rows = []
    for study, cfg in candidate_studies(root):
        key = f"{cfg.dataset.name}_seed{cfg.base_seed}"
        base = study.parent / "base"
        required = study_requirements(cfg) + ["native_features/train_complete.json"]
        missing = [name for name in required if not (study / name).exists()]
        for name in ("checkpoints/best.pt", "config.yaml", "metrics.json",
                     "history.csv", "artifacts/meta_targets.npz",
                     "artifacts/split_ids.npz"):
            if not (base / name).exists():
                missing.append("../base/" + name)
        reasons = []
        status, policy_digest, actual_digest = None, None, None
        try:
            metrics_path = study / "metrics.json"
            if metrics_path.exists():
                status = json.loads(metrics_path.read_text()).get("status")
            policy_path = study / "artifacts/policies_before_test.json"
            if policy_path.exists():
                policy_digest = json.loads(policy_path.read_text()).get("base_digest")
            if (base / "checkpoints/best.pt").exists():
                actual_digest = digest(base / "checkpoints/best.pt")
        except (OSError, ValueError) as exc:
            reasons.append(str(exc))
        if key not in EXPECTED:
            reasons.append("Not a requested base")
        elif cfg.base_digest != EXPECTED[key]:
            reasons.append("Study configuration digest differs from requested base")
        if missing:
            reasons.append("Required files missing")
        if status != "complete":
            reasons.append("Study status is not complete")
        if policy_digest != cfg.base_digest:
            reasons.append("Frozen policy digest differs from study")
        if actual_digest != cfg.base_digest:
            reasons.append("Checkpoint identity differs from study")
        rows.append(dict(
            key=key, study=str(study), base=str(base), status=status,
            expected_digest=EXPECTED.get(key), found_digest=cfg.base_digest,
            digest_matches=EXPECTED.get(key) == cfg.base_digest,
            policy_digest=policy_digest, checkpoint_digest=actual_digest,
            missing_files=missing, rejection_reasons=reasons, valid=not reasons,
        ))
    return dict(
        input_folders=[str(p) for p in sorted(root.iterdir())] if root.exists() else [],
        studies=rows, expected_jobs=list(EXPECTED),
    )


def discover(root):
    """Consume the exact same verified rows exposed by --inspect-inputs."""
    report = inspect_inputs(root)
    found = {}
    for row in report["studies"]:
        if not row["valid"]:
            continue
        key = row["key"]
        if key in found:
            raise ValueError(f"Duplicate full Outputs for {key}; attach one copy of each")
        found[key] = dict(key=key, base=row["base"], study=row["study"],
                          base_digest=row["checkpoint_digest"],
                          dataset=key.rsplit("_seed", 1)[0])
    missing = set(EXPECTED) - found.keys()
    if missing:
        raise ValueError(
            f"Missing complete Outputs: {sorted(missing)}. "
            "No experiment has started. Input diagnostics:\\n"
            + json.dumps(report, indent=2)
        )
    return [found[key] for key in EXPECTED]


def prepare_data(input_root, destination, jobs):
    """Link existing raw data read-only; do not copy large caches or redownload."""
    destination.mkdir(parents=True, exist_ok=True)
    required = []
    for job in jobs:
        required += (
            ["cifar-100-python", "CIFAR-100_human.pt"]
            if job["dataset"] == "cifar100n"
            else ["cifar-10-batches-py", "CIFAR-10_human.pt"]
        )
    for name in sorted(set(required)):
        candidates = sorted(Path(input_root).rglob(name))
        if not candidates:
            raise FileNotFoundError(f"Raw data absent from full Outputs: {name}")
        target = destination / name
        if not target.exists():
            target.symlink_to(candidates[0].resolve(), target_is_directory=candidates[0].is_dir())
    return destination


def export_batch(out, jobs, rows, status, started, commit):
    import csv
    config = dict(run_id=out.name, jobs=jobs, notebook_inputs=NOTEBOOKS,
                  git_commit=commit, base_retraining=False)
    OmegaConf.save(OmegaConf.create(config), out / "config.yaml")
    metrics = dict(status=status, stages=rows,
                   elapsed_seconds=time.monotonic() - started)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "run_metadata.json").write_text(json.dumps(config, indent=2))
    (out / "tables").mkdir(exist_ok=True)
    with (out / "tables/stage_status.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["job", "stage", "status", "seconds"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    text = "# Experiment Report: replication_followups\n\n## Configuration\n"
    text += f"Git: {commit}. No base training. Two exact checkpoint digests.\n\n"
    text += "## Dataset\nCIFAR-100N base44; CIFAR-10N Worse base43.\n\n"
    text += "## Checkpoint\nSee config.yaml for source paths and SHA256.\n\n"
    text += f"## Metrics\nStatus: {status}.\n\n## Compact Results\n"
    text += "\n".join(f"- {r['job']} / {r['stage']}: {r['status']}" for r in rows)
    text += "\n\n## Generated Artifacts\nPer-stage configs, metrics, reports and tables; full Outputs retain new heads/targets.\n"
    text += "\n## Notes\nRuntime is measured sequentially, without another GPU workload in this notebook. "
    text += "Intervals are conditional on saved models and policies, exploratory and uncorrected for multiplicity. "
    text += "Notebook URL order does not identify dataset; checkpoint digests do.\n"
    (out / "report.md").write_text(text)
    (out / "morning_summary.txt").write_text(
        f"Status: {status}\n" + "\n".join(
            f"{r['job']} {r['stage']}: {r['status']}" for r in rows
        )
    )
    with zipfile.ZipFile(out / "followup_summary.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(out)
            # Compact export excludes arrays, checkpoints and verbose logs.
            if (path.suffix in {".yaml", ".csv", ".md", ".txt", ".json"}
                    and ".hydra" not in relative.parts):
                archive.write(path, str(relative))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/rfd_followups"))
    parser.add_argument("--max-hours", type=float, default=6)
    parser.add_argument("--inspect-inputs", action="store_true",
                        help="Print attached study identities and missing files; no GPU or experiment")
    args = parser.parse_args()
    if args.inspect_inputs:
        print(json.dumps(inspect_inputs(args.input_root), indent=2))
        return
    if args.max_hours <= 0:
        raise ValueError("Positive time budget required")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU and Internet in Kaggle")
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    jobs = discover(args.input_root)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("Verified source bases:", json.dumps(jobs, indent=2), flush=True)
    data = prepare_data(args.input_root, out.parent / "rfd_followup_data", jobs)
    contract = dict(jobs=jobs, git_commit=commit, max_hours=args.max_hours,
                    bootstrap_draws=2000, augmentation_views=5, head_seeds=[42, 43, 44])
    contract_file = out / "batch_contract.yaml"
    if contract_file.exists():
        if OmegaConf.to_container(OmegaConf.load(contract_file)) != contract:
            raise ValueError("Existing batch scientific settings differ; use a new output")
    OmegaConf.save(OmegaConf.create(contract), contract_file)
    started = time.monotonic()
    rows = []
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg",
               MPLCONFIGDIR="/kaggle/working/matplotlib", OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4")
    export_batch(out, jobs, rows, "running", started, commit)
    # Cheap no-fit and runtime stages on both bases precede the new static heads.
    stages = [
        ("paired", "contrast-feature-targets"),
        ("runtime", "benchmark-runtime"),
        ("augmentation", "study-augmentation-control"),
    ]
    for stage, command in stages:
        for job in jobs:
            run_id = f"{job['key']}_{stage}"
            target = out / run_id
            metrics_file = target / "metrics.json"
            if metrics_file.exists() and json.loads(metrics_file.read_text()).get("status") == "complete":
                rows.append(dict(job=job["key"], stage=stage, status="complete_reused", seconds=0))
                continue
            remaining = args.max_hours - (time.monotonic() - started) / 3600
            if remaining < (0.5 if stage == "augmentation" else 0.1):
                rows.append(dict(job=job["key"], stage=stage,
                                 status="deferred_time_budget", seconds=0))
                export_batch(out, jobs, rows, "partial", started, commit)
                continue
            cmd = [sys.executable, str(root / "scripts/run.py"), command,
                   f"study_run={job['study']}", f"output_dir={out}", f"run_id={run_id}"]
            if stage != "paired":
                cmd += [f"source_run={job['base']}", f"data_dir={data}", "device=cuda"]
            if stage == "augmentation":
                cmd += [f"max_hours={min(2, remaining)}"]
            print(f"START {run_id}", flush=True)
            begin = time.monotonic()
            # Separate processes release CPU/GPU allocations between stages.
            with (out / f"{run_id}.log").open("w") as log:
                process = subprocess.Popen(cmd, cwd=root, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
            state = json.loads(metrics_file.read_text()).get("status") if metrics_file.exists() else "failed"
            if code:
                state = f"failed_exit_{code}"
            rows.append(dict(job=job["key"], stage=stage, status=state,
                             seconds=round(time.monotonic() - begin, 2)))
            export_batch(out, jobs, rows, "running", started, commit)
    complete = all(row["status"] in {"complete", "complete_reused"} for row in rows)
    export_batch(out, jobs, rows, "complete" if complete else "partial", started, commit)
    print((out / "morning_summary.txt").read_text(), flush=True)
    print("Download:", out / "followup_summary.zip", flush=True)
    if not complete:
        raise SystemExit("Some stages are incomplete; preserve full Output before retrying")


if __name__ == "__main__":
    main()
