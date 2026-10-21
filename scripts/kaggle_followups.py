"""Run bounded follow-up checks on recorded CIFAR bases, including external caches."""

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
REMAINING = {
    "cifar100n_seed42": "74a4b9d71a134bef4032afff26fa5f2d517ae5c769298980fca47033f6a7d355",
    "cifar100n_seed43": "2f6ff2f3ac7b26d28cd21ae21d5af50319acd3f9441ff1e154f39ee5c8190867",
    "cifar10n_seed42": "50f69a185caa1648775992f60035c5d0816f6f4ffe801b3edd685211a939bcae",
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
    ]
    for split in ("val", "test"):
        required += [
            f"native_features/{split}_complete.json",
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


def base_source(root, study, cfg):
    sibling = study.parent / "base"
    if (sibling / "checkpoints/best.pt").exists():
        return sibling
    hint = Path(str(cfg.get("source_run") or "/nonexistent-rfd-source"))
    if hint.is_relative_to(root) and (hint / "checkpoints/best.pt").exists():
        return hint
    matches = []
    for path in sorted(root.rglob("config.yaml")):
        if "heads" in path.parts or ".hydra" in path.parts:
            continue
        saved = OmegaConf.load(path)
        if (not OmegaConf.is_dict(saved) or not OmegaConf.is_dict(saved.get("model"))
                or saved.model.get("name") != "resnet50" or saved.get("epochs") != 50):
            continue
        checkpoint = path.parent / "checkpoints/best.pt"
        if checkpoint.exists() and digest(checkpoint) == cfg.base_digest:
            matches.append(path.parent)
    if len(matches) > 1:
        raise ValueError("Ambiguous base Outputs for " + cfg.base_digest)
    return matches[0] if matches else sibling


def cache_files(post):
    return [
        post / "cache" / name
        for split in ("val", "test")
        for name in ([f"{split}_complete.json"]
                     + [f"{split}_{suffix}.npy" for suffix in ("probs", "ids", "labels")])
    ]


def cache_source(root, study, cfg, cache_root=None):
    internal = study / "shared"
    if all(path.exists() for path in cache_files(internal)):
        return internal
    hint = Path(str(cfg.get("post_run") or "/nonexistent-rfd-cache"))
    if hint.is_relative_to(root) and all(path.exists() for path in cache_files(hint)):
        return hint
    matches = []
    markers = list(root.rglob("test_complete.json"))
    if cache_root is not None and Path(cache_root).exists():
        markers += list(Path(cache_root).rglob("test_complete.json"))
    for marker in sorted(set(markers)):
        if marker.parent.name != "cache":
            continue
        post = marker.parent.parent
        if not all(p.exists() for p in cache_files(post)):
            continue
        specs = [json.loads((post / "cache" / f"{split}_complete.json").read_text())
                 for split in ("val", "test")]
        if all(
            spec.get("base_digest") == cfg.base_digest
            and spec.get("seed") == cfg.inference_seed
            and spec.get("views", 0) >= max(cfg.view_counts)
            for spec in specs
        ):
            matches.append(post)
    if len(matches) > 1:
        raise ValueError("Ambiguous shared cache Outputs for " + cfg.base_digest)
    return matches[0] if matches else internal


def inspect_inputs(root, expected=None, cache_root=None):
    """Run the same full preflight used by execution, without a GPU or training."""
    root = Path(root).resolve()
    expected = EXPECTED if expected is None else expected
    rows = []
    for study, cfg in candidate_studies(root):
        key = f"{cfg.dataset.name}_seed{cfg.base_seed}"
        reasons = []
        if key in expected:
            try:
                base = base_source(root, study, cfg)
                post = cache_source(root, study, cfg, cache_root)
            except ValueError as exc:
                base, post = study.parent / "base", study / "shared"
                reasons.append(str(exc))
        else:
            base, post = study.parent / "base", study / "shared"
        required = study_requirements(cfg) + ["native_features/train_complete.json"]
        missing = [name for name in required if not (study / name).exists()]
        for name in ("checkpoints/best.pt", "config.yaml", "metrics.json",
                     "history.csv", "artifacts/meta_targets.npz",
                     "artifacts/split_ids.npz"):
            if not (base / name).exists():
                missing.append("../base/" + name)
        missing += [str(path) for path in cache_files(post) if not path.exists()]
        status, policy_digest, actual_digest = None, None, None
        try:
            metrics_path = study / "metrics.json"
            if metrics_path.exists():
                status = json.loads(metrics_path.read_text()).get("status")
            for split in ("val", "test"):
                marker = post / "cache" / f"{split}_complete.json"
                if marker.exists():
                    protocol = json.loads(marker.read_text())
                    if (protocol.get("base_digest") != cfg.base_digest
                            or protocol.get("seed") != cfg.get("inference_seed", 20261015)
                            or protocol.get("views", 0) < max(cfg.get("view_counts", [1]))):
                        reasons.append(f"{split} shared cache provenance differs from study")
            policy_path = study / "artifacts/policies_before_test.json"
            if policy_path.exists():
                policy_digest = json.loads(policy_path.read_text()).get("base_digest")
            if (base / "checkpoints/best.pt").exists():
                actual_digest = digest(base / "checkpoints/best.pt")
        except (OSError, ValueError) as exc:
            reasons.append(str(exc))
        if key not in expected:
            reasons.append("Not a requested base")
        elif cfg.base_digest != expected[key]:
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
            key=key, study=str(study), base=str(base), post=str(post), status=status,
            expected_digest=expected.get(key), found_digest=cfg.base_digest,
            digest_matches=expected.get(key) == cfg.base_digest,
            policy_digest=policy_digest, checkpoint_digest=actual_digest,
            missing_files=missing, rejection_reasons=reasons, valid=not reasons,
        ))
    return dict(
        input_folders=[str(p) for p in sorted(root.iterdir())] if root.exists() else [],
        studies=rows, expected_jobs=list(expected),
    )


def discover(root, expected=None, cache_root=None):
    """Consume the exact same verified rows exposed by --inspect-inputs."""
    expected = EXPECTED if expected is None else expected
    report = inspect_inputs(root, expected, cache_root)
    found = {}
    for row in report["studies"]:
        if not row["valid"]:
            continue
        key = row["key"]
        if key in found:
            raise ValueError(f"Duplicate full Outputs for {key}; attach one copy of each")
        found[key] = dict(key=key, base=row["base"], study=row["study"], post=row["post"],
                          base_digest=row["checkpoint_digest"],
                          dataset=key.rsplit("_seed", 1)[0])
    missing = set(expected) - found.keys()
    if missing:
        raise ValueError(
            f"Missing complete Outputs: {sorted(missing)}. "
            "No experiment has started. Input diagnostics:\\n"
            + json.dumps(report, indent=2)
        )
    return [found[key] for key in expected]


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
        if target.is_symlink() and not target.exists():
            target.unlink()
        if not target.exists():
            target.symlink_to(candidates[0].resolve(), target_is_directory=candidates[0].is_dir())
    return destination


def export_batch(out, jobs, rows, status, started, commit, profile="replications"):
    import csv
    config = dict(run_id=out.name, jobs=jobs, profile=profile,
                  notebook_inputs=NOTEBOOKS if profile == "replications" else [],
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
    text = f"# Experiment Report: {out.name}\n\n## Configuration\n"
    text += f"Git: {commit}. No base training. {len(jobs)} exact checkpoint digests.\n\n"
    text += "## Dataset\n" + "; ".join(job["key"] for job in jobs) + ".\n\n"
    text += "## Checkpoint\nSee config.yaml for source paths and SHA256.\n\n"
    text += f"## Metrics\nStatus: {status}.\n\n## Compact Results\n"
    text += "\n".join(f"- {r['job']} / {r['stage']}: {r['status']}" for r in rows)
    text += "\n\n## Generated Artifacts\nPer-stage configs, metrics, reports and tables; full Outputs retain new heads/targets.\n"
    text += "\n## Notes\n" + ("Runtime is measured sequentially. " if profile == "replications" else "Runtime is not repeated in this control-only profile. ")
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=["replications", "remaining"],
                        default="replications")
    parser.add_argument("--max-hours", type=float, default=6)
    parser.add_argument("--jobs", nargs="+", help="Explicit subset of profile jobs; recorded in batch config")
    parser.add_argument("--cache-root", type=Path, help="Additional local recovered-cache directory")
    parser.add_argument("--inspect-inputs", action="store_true",
                        help="Print attached study identities and missing files; no GPU or experiment")
    args = parser.parse_args()
    expected = EXPECTED if args.profile == "replications" else REMAINING
    if args.jobs:
        unknown = set(args.jobs) - expected.keys()
        if unknown or len(set(args.jobs)) != len(args.jobs):
            parser.error("Unknown or duplicate jobs: " + str(args.jobs))
        expected = {key: expected[key] for key in args.jobs}
    if args.output is None:
        args.output = Path("/kaggle/working/rfd_followups" if args.profile == "replications"
                           else "/kaggle/working/rfd_cifar_controls_remaining")
    if args.inspect_inputs:
        print(json.dumps(inspect_inputs(args.input_root, expected, args.cache_root), indent=2))
        return
    if args.max_hours <= 0:
        raise ValueError("Positive time budget required")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU and Internet in Kaggle")
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    jobs = discover(args.input_root, expected, args.cache_root)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("Verified source bases:", json.dumps(jobs, indent=2), flush=True)
    data = prepare_data(args.input_root, out.parent / "rfd_followup_data", jobs)
    contract = dict(jobs=jobs, profile=args.profile, git_commit=commit, max_hours=args.max_hours,
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
    export_batch(out, jobs, rows, "running", started, commit, args.profile)
    # No-fit contrasts (and optional runtime) precede new static heads.
    stages = [
        ("paired", "contrast-feature-targets"),
        ("runtime", "benchmark-runtime"),
        ("augmentation", "study-augmentation-control"),
    ]
    if args.profile == "remaining":
        stages = [(stage, command) for stage, command in stages if stage != "runtime"]
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
                export_batch(out, jobs, rows, "partial", started, commit, args.profile)
                continue
            cmd = [sys.executable, str(root / "scripts/run.py"), command,
                   f"study_run={job['study']}", f"output_dir={out}", f"run_id={run_id}"]
            if stage != "runtime":
                cmd += [f"post_run={job['post']}"]
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
            export_batch(out, jobs, rows, "running", started, commit, args.profile)
    complete = all(row["status"] in {"complete", "complete_reused"} for row in rows)
    export_batch(out, jobs, rows, "complete" if complete else "partial", started, commit, args.profile)
    print((out / "morning_summary.txt").read_text(), flush=True)
    print("Download:", out / "followup_summary.zip", flush=True)
    if not complete:
        raise SystemExit("Some stages are incomplete; preserve full Output before retrying")


if __name__ == "__main__":
    main()
