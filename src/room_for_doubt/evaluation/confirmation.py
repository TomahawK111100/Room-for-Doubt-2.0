"""Read-only external evaluation of three verified, already frozen S4 runs."""

import csv
import gc
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.datasets import CIFAR10
from room_for_doubt.data.cifar10_1 import load_v6
from room_for_doubt.data.cifar_noisy import load_observed_labels
from room_for_doubt.data.confirmation import (
    CLASSES,
    METHODS,
    SEEDS,
    verify_frozen,
    write_json,
)
from room_for_doubt.data.kaggle_artifacts import load_completed_run
from room_for_doubt.data.observed_images import imagenet_transform
from room_for_doubt.evaluation.diagnostic_metrics import probability_metrics
from room_for_doubt.evaluation.tie_aware_risk import expected_tie_aurc
from room_for_doubt.inference.confirmation import make_confirmation_plan
from room_for_doubt.reporting.artifacts import write_csv
from room_for_doubt.training.temporal_snapshots import file_digest, verify_snapshots


def paired_stratified_interval(differences, labels, repeats=10_000):
    """Resample the same IDs across the fixed bases, preserving class counts."""
    per_image = np.asarray(differences, dtype=float).mean(axis=0)
    labels = np.asarray(labels)
    if per_image.shape != labels.shape:
        raise ValueError("Paired differences must align by image ID")
    rng = np.random.default_rng(20261015)
    estimates = np.zeros(repeats, dtype=float)
    for label in np.unique(labels):
        group = per_image[labels == label]
        for start in range(0, repeats, 256):
            count = min(256, repeats - start)
            ids = rng.integers(0, len(group), (count, len(group)))
            estimates[start : start + count] += group[ids].sum(axis=1)
    return (np.quantile(estimates / len(labels), [0.025, 0.975]) * 100).tolist()


def runtime_summary(roots):
    rows = []
    for seed in SEEDS:
        with (Path(roots[seed]) / "runtime/tables/runtime.csv").open() as handle:
            rows.extend(
                row
                for row in csv.DictReader(handle)
                if int(row["seed"]) == seed
                and row["batch_size"] == "1"
                and row["scope"] == "pipeline"
            )
    return {
        method: (
            mean(float(r["median_ms_per_batch"]) for r in rows if r["method"] == method)
            if all(
                sum(
                    r["method"] == method
                    and int(r["seed"]) == seed
                    and int(r["blocks"]) == 10
                    and int(r["oom_blocks"]) == 0
                    for r in rows
                )
                == 1
                for seed in SEEDS
            )
            else None
        )
        for method in METHODS
    }


@torch.inference_mode()
def predict(root, bundle, images, method, device):
    plan, _ = make_confirmation_plan(root, bundle, method, device)
    return predict_batches(plan, images, device)


@torch.inference_mode()
def predict_batches(plan, images, device):
    transform = imagenet_transform(224)
    batches = []
    for start in range(0, len(images), 32):
        x = torch.stack(
            [transform(Image.fromarray(image)) for image in images[start : start + 32]]
        )
        batches.append(plan(x.to(device)).cpu().numpy())
    return np.concatenate(batches)


def evaluate(roots, train_data, external, output, protocol_hash, device):
    if set(roots) != set(SEEDS):
        raise ValueError(
            "Require all three frozen S4 seeds before accessing external data"
        )
    manifests = {
        seed: verify_frozen(roots[seed], seed, protocol_hash) for seed in SEEDS
    }
    if len({m["files"]["base/checkpoints/best.pt"] for m in manifests.values()}) != 3:
        raise ValueError("Expected three independently trained base checkpoints")
    contract = {
        "protocol_sha256": protocol_hash,
        "frozen_manifest_sha256": {
            str(seed): file_digest(Path(roots[seed]) / "frozen.json") for seed in SEEDS
        },
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    config = out / "evaluation_contract.json"
    if config.exists():
        if json.loads(config.read_text()) != contract:
            raise ValueError("Existing evaluation belongs to another frozen cohort")
        if (out / "metrics.json").exists():
            raise ValueError("Evaluation already complete. Use saved results.")
    write_json(config, contract)
    raw = CIFAR10(str(train_data), train=True, download=True)
    if tuple(raw.classes) != CLASSES:
        raise ValueError("Unexpected training class order")
    noisy = load_observed_labels(str(train_data), "cifar10n", "worse")
    bundles = {
        seed: load_completed_run(Path(roots[seed]) / "base", noisy) for seed in SEEDS
    }
    for bundle in bundles.values():
        verify_snapshots(bundle["source"], bundle["config"], [1, 2, 3, 4, 5])
    images, labels, digests = load_v6(external, raw.data)
    OmegaConf.save(
        OmegaConf.create(
            dict(
                **contract,
                dataset="CIFAR-10.1 v6",
                data_sha256=digests,
                base_seeds=list(SEEDS),
                predictors=list(METHODS),
                device=str(device),
                bootstrap_seed=20261015,
                bootstrap_repeats=10_000,
            )
        ),
        out / "config.yaml",
        resolve=True,
    )
    rows, correctness = [], {name: [] for name in METHODS}
    for seed, bundle in bundles.items():
        saved = {}
        for method in METHODS:
            probs = predict(roots[seed], bundle, images, method, device)
            correct = probs.argmax(1) == labels
            correctness[method].append(correct.astype(float))
            rows.append(
                dict(
                    seed=seed,
                    method=method,
                    **probability_metrics(labels, probs),
                    aurc=expected_tie_aurc(probs.max(1), ~correct),
                )
            )
            saved[method] = probs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        np.savez_compressed(
            out / f"seed{seed}_predictions.npz",
            ids=np.arange(len(labels)),
            labels=labels,
            **saved,
        )
    delta = np.array(correctness["early5"]) - correctness["base"]
    gap = np.array(correctness["early5_teacher"]) - correctness["early5"]
    delta_ci = paired_stratified_interval(delta, labels)
    latency = runtime_summary(roots)
    gains = (delta.mean(axis=1) * 100).tolist()
    gap_pp = float(gap.mean() * 100)
    ratios = (
        {
            "student_to_base": latency["early5"] / latency["base"],
            "student_to_teacher": latency["early5"] / latency["early5_teacher"],
        }
        if all(latency[name] for name in ("early5", "base", "early5_teacher"))
        else None
    )
    metrics = dict(
        status="complete",
        dataset="CIFAR-10.1 v6",
        examples=len(labels),
        base_gain_pp=gains,
        mean_gain_pp=mean(gains),
        gain_ci95_pp=delta_ci,
        gain_between_base_sd_pp=float(np.std(gains, ddof=1)),
        teacher_gap_pp=gap_pp,
        teacher_gap_ci95_pp=paired_stratified_interval(gap, labels),
        pipeline_batch1_ms=latency,
        latency_ratios=ratios,
        quality_pass=mean(gains) >= 0.5 and delta_ci[0] > 0 and gap_pp <= 2,
        latency_pass=(
            (ratios["student_to_base"] <= 1.10 and ratios["student_to_teacher"] <= 0.50)
            if ratios is not None
            else None
        ),
    )
    write_csv(out / "metrics.csv", rows)
    write_json(
        out / "run_metadata.json",
        dict(**contract, data_sha256=digests, device=str(device)),
    )
    (out / "report.md").write_text(
        "# S4 external evaluation\n\n"
        "CIFAR-10.1 v6. Three frozen bases, four predictors per base.\n\n"
        "See metrics.csv and metrics.json for test results and validation runtime. "
        "Paired stratified intervals are conditional on the fitted bases.\n"
    )
    write_json(out / "metrics.json", metrics)
    return metrics
