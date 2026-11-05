"""Predeclared independent-base replication plus fixed epoch-1 distillation control."""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
from omegaconf import OmegaConf
from room_for_doubt.data.temporal_evidence import load_evidence, seal_evidence
from room_for_doubt.evaluation.selection import paired_accuracy_interval
from room_for_doubt.training.temporal_snapshots import file_digest

ROOT = Path(__file__).resolve().parents[1]


def single_snapshot_cache(source, target):
    """Reuse identical inputs/partitions; only replace the teacher distribution."""
    source, target = Path(source), Path(target)
    manifest, _ = load_evidence(source)
    snapshot = [s for s in manifest["snapshots"] if s["epoch"] == 1]
    if len(snapshot) != 1 or target.exists():
        raise ValueError("Require real epoch1 evidence and a fresh control cache")
    target.mkdir(parents=True)
    for name, item in manifest["arrays"].items():
        if name in ("teacher_probs", "val_teacher_probs"):
            prefix = "train" if name == "teacher_probs" else "val"
            q = np.array(
                np.load(source / f"{prefix}_epoch001_probs.npy", allow_pickle=False),
                dtype=np.float32,
            )
            q /= q.sum(1, keepdims=True)
            np.save(target / item["file"], q)
        else:
            # Never modify a linked source array; control heads only read evidence.
            os.link(source / item["file"], target / item["file"])
    metadata = {
        k: v for k, v in manifest.items() if k not in ("arrays", "schema", "scope")
    }
    metadata.update(
        snapshots=snapshot, acquisition="canonical eval FP32; fixed epoch1 teacher only"
    )
    return seal_evidence(target, metadata)


def execute(out, data, seed):
    out, data = Path(out).resolve(), Path(data).resolve()
    if out.exists() and (out / "config.yaml").exists():
        raise ValueError("Preserve existing run; launch only in a fresh session")
    out.mkdir(parents=True, exist_ok=True)
    base_id = f"temporal_cifar100n_seed{seed}_v1"
    source = out / "bases" / base_id
    protocol = dict(
        seed=seed,
        head_seed=42,
        base_split_seed=42,
        base_epochs=50,
        base_max_hours=4.0,
        head_epochs=25,
        hidden_dim=256,
        branches=dict(temporal=[1, 2, 3, 4, 5], single_epoch1=[1]),
        alphas=[0.0, 0.25, 0.5, 1.0],
        selection="noisy validation only",
        final_test="automatic only after both policies frozen",
        control_scope="same base/features/meta split/head seed/optimizer/epochs; single fixed epoch1 teacher, not a compute-matched ensemble",
    )
    OmegaConf.save(OmegaConf.create(protocol), out / "config.yaml")
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

    def run(name, args):
        print("START", name, flush=True)
        (out / "progress.json").write_text(
            json.dumps(dict(stage=name, status="running"))
        )
        with (out / (name + ".log")).open("w") as log:
            result = subprocess.run(
                driver + args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect saved log")
        print("DONE", name, flush=True)

    branch_info = {}
    try:
        run(
            "base",
            [
                "train-kaggle-base",
                "--config-name",
                "temporal_base",
                f"run_id={base_id}",
                f"seed={seed}",
                f'output_dir={out/"bases"}',
                f"data_dir={data}",
                "num_workers=0",
                "max_hours=4.0",
            ],
        )
        base = json.loads((source / "metrics.json").read_text())
        if base["status"] != "complete":
            raise RuntimeError(
                "Base stopped at time budget; checkpoint preserved, no downstream/test run"
            )
        for name, epochs in protocol["branches"].items():
            head_id = f"{name}_base{seed}_head42"
            cache = out / ("evidence_" + name)
            common = [
                "distill-temporal",
                f"run_id={head_id}",
                f"source_run={source}",
                f"data_dir={data}",
                f'output_dir={out/"heads"}',
                f"cache_dir={cache}",
                "seed=42",
                "teacher_epochs=[" + ",".join(map(str, epochs)) + "]",
                "device=cuda",
            ]
            if name == "temporal":
                run("prepare_" + name, common + ["stage=prepare"])
            else:
                single_snapshot_cache(out / "evidence_temporal", cache)
            run("fit_" + name, common + ["stage=fit"])
            policy = out / "heads" / head_id / "artifacts/policy_before_test.json"
            branch_info[name] = dict(
                run_id=head_id, policy_sha256=file_digest(policy), command=common
            )
        # Both branches are irreversibly selected before either touches test labels.
        (out / "policies_before_any_test.json").write_text(
            json.dumps(branch_info, indent=2)
        )
        results = {}
        for name, info in branch_info.items():
            policy = (
                out / "heads" / info["run_id"] / "artifacts/policy_before_test.json"
            )
            if file_digest(policy) != info["policy_sha256"]:
                raise ValueError("Frozen policy changed before test")
            run("evaluate_" + name, info["command"] + ["stage=evaluate"])
            directory = out / "heads" / info["run_id"] / "final_evaluation"
            results[name] = json.loads((directory / "metrics.json").read_text())
        with (
            np.load(
                out
                / "heads"
                / branch_info["temporal"]["run_id"]
                / "final_evaluation/predictions.npz"
            ) as a,
            np.load(
                out
                / "heads"
                / branch_info["single_epoch1"]["run_id"]
                / "final_evaluation/predictions.npz"
            ) as b,
        ):
            if (
                not np.array_equal(a["ids"], b["ids"])
                or not np.array_equal(a["labels"], b["labels"])
                or not np.array_equal(a["base"], b["base"])
            ):
                raise ValueError("Branch test IDs, labels or base probabilities differ")
            paired = paired_accuracy_interval(
                a["labels"], b["corrected"].argmax(1), a["corrected"].argmax(1)
            )
        metrics = dict(
            status="complete",
            base_seed=seed,
            head_seed=42,
            branches=results,
            temporal_minus_single_epoch1=paired,
            independent_bases=1,
            paired_scope="nominal paired-example interval, conditional on one fitted base; no multiplicity adjustment",
        )
    except Exception as e:
        metrics = dict(
            status="failed", error=str(e), base_seed=seed, branches_frozen=branch_info
        )
        raise
    finally:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (out / "progress.json").write_text(json.dumps(dict(status=metrics["status"])))
        (out / "run_metadata.json").write_text(
            json.dumps(
                dict(
                    base_seed=seed,
                    head_seed=42,
                    protocol=protocol,
                    policies=branch_info,
                ),
                indent=2,
            )
        )
        (out / "tables").mkdir(exist_ok=True)
        rows = []
        for branch, result in metrics.get("branches", {}).items():
            for method, values in result["methods"].items():
                rows.append(
                    dict(
                        branch=branch,
                        method=method,
                        **{
                            k: (None if v is None else float(f"{v:.4g}"))
                            for k, v in values.items()
                        },
                    )
                )
        if rows:
            with (out / "tables/methods.csv").open("w", newline="") as h:
                w = csv.DictWriter(h, fieldnames=list(rows[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
        text = f"Status: {metrics['status']}\nBase seed: {seed}\nHead seed: 42\n"
        for r in rows:
            text += f"{r['branch']} {r['method']}: acc={r['acc']}\n"
        (out / "morning_summary.txt").write_text(text)
        (out / "report.md").write_text(
            f'# Experiment Report: {out.name}\n\n## Configuration\nSee config.yaml.\n\n## Dataset\nCIFAR100N noisy100, fixed base split42.\n\n## Checkpoint\nBase minimum noisy-validation CE; each head minimum meta-validation KL.\n\n## Metrics\nSee metrics.json.\n\n## Compact Results\nSee tables/methods.csv.\n\n## Generated Artifacts\nSource, base checkpoints, teacher evidence, both heads, frozen policies, raw final predictions.\n\n## Notes\n{metrics.get("error", "One independent base; single epoch1 control is not a compute-matched ensemble. No test-based selection.")}\n'
        )
        for full in (False, True):
            filename = "replication_results.zip" if full else "replication_summary.zip"
            with zipfile.ZipFile(out / filename, "w", zipfile.ZIP_DEFLATED) as z:
                for p in sorted(out.rglob("*")):
                    if (
                        p.is_file()
                        and (
                            p.suffix
                            in {".json", ".yaml", ".csv", ".md", ".txt", ".log"}
                            or (full and p.name == "predictions.npz")
                        )
                        and not any(
                            x.startswith("evidence_") for x in p.relative_to(out).parts
                        )
                    ):
                        z.write(p, str(p.relative_to(out)))
        print(text, flush=True)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=[43, 44], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--data-dir", required=True)
    args = p.parse_args()
    execute(args.output, args.data_dir, args.seed)
