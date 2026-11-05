"""Post-pilot validation-selected single-snapshot control; never refit a base."""

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
from room_for_doubt.data.temporal_evidence import (
    load_evidence,
    seal_evidence,
    probabilities,
)
from room_for_doubt.evaluation.temporal_residual import corrected_probabilities
from room_for_doubt.evaluation.diagnostic_metrics import probability_metrics
from room_for_doubt.evaluation.selection import paired_accuracy_interval
from room_for_doubt.training.temporal_snapshots import file_digest

ROOT = Path(__file__).resolve().parents[1]


def choose_epoch(base, candidates, noisy_labels):
    rows = []
    for epoch, q in sorted(candidates.items()):
        probabilities(q)
        h = np.log(np.clip(q, 1e-12, 1)) - np.log(np.clip(base, 1e-12, 1))
        for alpha in [0.0, 0.25, 0.5, 1.0]:
            m = probability_metrics(
                noisy_labels, corrected_probabilities(base, h, alpha)
            )
            rows.append(dict(epoch=epoch, alpha=alpha, acc=m["acc"], nll=m["nll"]))
    # Predeclared deterministic ties: accuracy, NLL, smaller alpha, earlier epoch.
    selected = max(rows, key=lambda r: (r["acc"], -r["nll"], -r["alpha"], -r["epoch"]))
    return selected, rows


def control_cache(evidence, target, epoch):
    evidence, target = Path(evidence), Path(target)
    m, a = load_evidence(evidence)
    if target.exists():
        raise ValueError("Fresh control cache required")
    snaps = [s for s in m["snapshots"] if s["epoch"] == epoch]
    if len(snaps) != 1:
        raise ValueError("Selected snapshot is absent")
    target.mkdir(parents=True)
    for name, item in m["arrays"].items():
        if name in ("teacher_probs", "val_teacher_probs"):
            split = "train" if name == "teacher_probs" else "val"
            q = np.array(
                np.load(
                    evidence / f"{split}_epoch{epoch:03d}_probs.npy", allow_pickle=False
                ),
                dtype=np.float32,
            )
            q /= q.sum(1, keepdims=True)
            np.save(target / item["file"], q)
        else:
            shutil.copyfile(evidence / item["file"], target / item["file"])
    meta = {k: v for k, v in m.items() if k not in ("schema", "scope", "arrays")}
    meta.update(
        snapshots=snaps,
        acquisition="canonical eval FP32; noisy-validation selected single snapshot",
    )
    return seal_evidence(target, meta)


def locate(inputs):
    found = {}
    for manifest in Path(inputs).rglob("manifest.json"):
        if manifest.parent.name not in ("evidence", "evidence_temporal"):
            continue
        root = manifest.parent.parent
        bases = list((root / "bases").glob("temporal_cifar100n_seed*_v1"))
        if len(bases) != 1:
            continue
        cfg = OmegaConf.to_container(
            OmegaConf.load(bases[0] / "config.yaml"), resolve=True
        )
        seed = int(cfg["seed"])
        if seed not in (43, 44):
            continue
        head = (
            root
            / "heads"
            / (
                f"temporal_base{seed}_head42"
                if seed != 42
                else "temporal_residual_seed42_v1"
            )
        )
        required = [
            head / "artifacts/policy_before_test.json",
            bases[0] / "checkpoints/best.pt",
            head / "final_evaluation/predictions.npz",
            head / "final_evaluation/metrics.json",
        ]
        required += [
            bases[0] / name
            for name in (
                "metrics.json",
                "history.csv",
                "artifacts/meta_targets.npz",
                "artifacts/split_ids.npz",
                "checkpoints/temporal/manifest.json",
            )
        ]
        required += [
            bases[0] / f"checkpoints/temporal/epoch_{e:03d}.pt" for e in range(1, 6)
        ]
        required += [
            manifest.parent / f"{split}_epoch{e:03d}_probs.npy"
            for split in ("train", "val")
            for e in range(1, 6)
        ]
        if not all(p.is_file() for p in required):
            raise ValueError(
                f"Incomplete full Output for seed{seed}; missing {[str(p) for p in required if not p.is_file()]}"
            )
        load_evidence(manifest.parent)
        old = json.loads((head / "artifacts/policy_before_test.json").read_text())
        if (
            file_digest(manifest.parent / "manifest.json")
            != old["cache_manifest_sha256"]
            or file_digest(bases[0] / "checkpoints/best.pt") != old["base_digest"]
        ):
            raise ValueError("Source base/cache differ from original frozen policy")
        if seed in found:
            raise ValueError("Duplicate complete source base")
        found[seed] = dict(
            root=root, base=bases[0], evidence=manifest.parent, head=head
        )
    if set(found) != {43, 44}:
        raise ValueError(
            f"Attach full replications43/44 Outputs; found {sorted(found)}"
        )
    return found


def execute(inputs, out, data):
    out = Path(out)
    data = Path(data)
    if (out / "config.yaml").exists():
        raise ValueError("Fresh output required")
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        seeds=[43, 44],
        candidate_epochs=[1, 2, 3, 4, 5],
        teacher_alpha_grid=[0.0, 0.25, 0.5, 1.0],
        head_seed=42,
        head_epochs=25,
        selection="noisy-validation accuracy, NLL, smaller alpha, earlier epoch",
        test_scope="post-pilot reuse of known test set; exploratory",
        base_training=False,
    )
    OmegaConf.save(OmegaConf.create(protocol), out / "config.yaml")
    driver = [sys.executable, str(ROOT / "scripts/run.py")]
    env = dict(
        os.environ,
        PYTHONPATH=str(ROOT / "src"),
        PYTHONUNBUFFERED="1",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(out / "matplotlib"),
    )
    sources = locate(inputs)
    policies = {}
    metrics = dict(status="running", protocol=protocol)

    def run(name, args):
        print("START", name, flush=True)
        with (out / (name + ".log")).open("w") as f:
            r = subprocess.run(
                driver + args, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT
            )
        if r.returncode:
            raise RuntimeError(f"{name} failed; inspect preserved log")

    try:
        for seed, s in sources.items():
            manifest, arrays = load_evidence(s["evidence"])
            old = json.loads(
                (s["head"] / "artifacts/policy_before_test.json").read_text()
            )
            if (
                file_digest(s["evidence"] / "manifest.json")
                != old["cache_manifest_sha256"]
            ):
                raise ValueError("Source cache differs from frozen temporal head")
            candidates = {
                e: np.load(
                    s["evidence"] / f"val_epoch{e:03d}_probs.npy", allow_pickle=False
                )
                for e in range(1, 6)
            }
            # Verify retained per-epoch arrays reconstruct the sealed original teacher.
            q = np.mean(
                np.stack([candidates[e] for e in range(1, 6)]).astype(np.float64),
                axis=0,
            ).astype(np.float32)
            q /= q.sum(1, keepdims=True)
            if not np.array_equal(q, arrays["val_teacher_probs"]):
                raise ValueError(
                    "Per-epoch val probabilities do not reconstruct sealed teacher"
                )
            chosen, rows = choose_epoch(
                arrays["val_probs"], candidates, arrays["val_noisy_labels"]
            )
            epoch = chosen["epoch"]
            cache = out / f"evidence_seed{seed}"
            control_cache(s["evidence"], cache, epoch)
            name = f"selected_single_base{seed}_head42"
            head = out / "heads" / name
            common = [
                "distill-temporal",
                f"run_id={name}",
                f'source_run={s["base"]}',
                f"data_dir={data}",
                f'output_dir={out/"heads"}',
                f"cache_dir={cache}",
                f"teacher_epochs=[{epoch}]",
                "seed=42",
                "device=cuda",
            ]
            with (out / f"selection_seed{seed}.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
            run(f"fit_seed{seed}", common + ["stage=fit"])
            policies[seed] = dict(
                epoch=epoch,
                teacher_selection=chosen,
                policy_sha256=file_digest(head / "artifacts/policy_before_test.json"),
                command=common,
                head=str(head),
                source_head=str(s["head"]),
            )
        (out / "all_policies_before_test.json").write_text(
            json.dumps(policies, indent=2)
        )
        results = {}
        for seed, p in policies.items():
            head = Path(p["head"])
            if (
                file_digest(head / "artifacts/policy_before_test.json")
                != p["policy_sha256"]
            ):
                raise ValueError("Control policy changed")
            run(f"evaluate_seed{seed}", p["command"] + ["stage=evaluate"])
            result = json.loads((head / "final_evaluation/metrics.json").read_text())
            # Reuse saved temporal test predictions; do not retune original method.
            original = Path(p["source_head"]) / "final_evaluation/predictions.npz"
            if not original.is_file():
                raise ValueError("Missing original temporal test predictions")
            with (
                np.load(original, allow_pickle=False) as a,
                np.load(
                    head / "final_evaluation/predictions.npz", allow_pickle=False
                ) as b,
            ):
                if not all(
                    np.array_equal(a[k], b[k]) for k in ("ids", "labels", "base")
                ):
                    raise ValueError("Matched test inputs differ")
                pair = paired_accuracy_interval(
                    a["labels"], b["corrected"].argmax(1), a["corrected"].argmax(1)
                )
            results[seed] = dict(
                selected_epoch=p["epoch"],
                control=result,
                temporal_minus_selected_single=pair,
            )
        metrics = dict(status="complete", protocol=protocol, results=results)
    except Exception as e:
        metrics = dict(status="failed", error=str(e), protocol=protocol)
        raise
    finally:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (out / "run_metadata.json").write_text(
            json.dumps(dict(protocol=protocol, policies=policies), indent=2)
        )
        (out / "morning_summary.txt").write_text(
            "Status: "
            + metrics["status"]
            + "\n"
            + json.dumps(metrics.get("results", metrics.get("error")), indent=2)
        )
        (out / "report.md").write_text(
            "# Experiment Report: selected single-snapshot control\n\n## Configuration\nSee config.yaml.\n\n## Dataset\nCIFAR100N; known clean test reused only after noisy-validation selection.\n\n## Checkpoint\nHead minimum meta-validation KL; original base unchanged.\n\n## Metrics\nSee metrics.json.\n\n## Compact Results\nSee selection_seed*.csv.\n\n## Generated Artifacts\nPolicies, selection tables, predictions, source and logs.\n\n## Notes\nExploratory post-pilot control; no independent test confirmation claimed.\n"
        )
        for full in (False, True):
            with zipfile.ZipFile(
                out
                / (
                    "selected_control_results.zip"
                    if full
                    else "selected_control_summary.zip"
                ),
                "w",
                zipfile.ZIP_DEFLATED,
            ) as z:
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--input-root", default="/kaggle/input")
    a = p.parse_args()
    execute(a.input_root, a.output, a.data_dir)
