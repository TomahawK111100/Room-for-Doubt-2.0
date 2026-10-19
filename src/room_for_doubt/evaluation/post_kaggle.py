"""Same-view comparison with all noisy-validation selections before test metrics."""

import json
from pathlib import Path
import numpy as np
from room_for_doubt.evaluation.corrected_campaign import predict_arrays
from room_for_doubt.evaluation.trajectory_scoring import mean_margin_badness
from room_for_doubt.evaluation.selection import (
    select_temperature,
    weighted_consensus,
    error_metrics,
    paired_accuracy_interval,
)
from room_for_doubt.training.post_kaggle_meta import load_head
from room_for_doubt.reporting.artifacts import write_csv


def trajectory_scores(model, features, probs, cfg, device):
    n, k = probs.shape[:2]
    paths = predict_arrays(
        model, features.reshape(-1, 512), probs.argmax(2).reshape(-1), cfg, device
    )
    return mean_margin_badness(paths).reshape(n, k)


def freeze_policies(cfg, bundle, root, device):
    root = Path(root)
    cache = root / "cache"
    artifacts = root / "artifacts"
    probs = np.load(cache / "val_probs.npy", mmap_mode="r")
    features = np.load(cache / "val_features.npy", mmap_mode="r")
    labels = np.load(cache / "val_labels.npy")
    ids = np.load(cache / "val_ids.npy")
    if not np.array_equal(ids, bundle["data"]["base_val_ids"]):
        raise ValueError("Validation membership mismatch")
    policies = []
    candidates = [("msp", None, None)] + [
        (mode, seed, load_head(cfg, bundle, mode, seed, root, device))
        for mode in cfg.modes
        for seed in cfg.seeds
    ]
    for mode, seed, model in candidates:
        badness = (
            1 - probs.max(2)
            if mode == "msp"
            else trajectory_scores(model, features, probs, cfg, device)
        )
        temperature, sweep = select_temperature(
            probs, badness, labels, cfg.temperatures
        )
        name = "msp" if mode == "msp" else f"{mode}_{seed}"
        np.save(artifacts / f"val_badness_{name}.npy", badness)
        policies.append(
            dict(name=name, mode=mode, seed=seed, temperature=temperature, sweep=sweep)
        )
    frozen = dict(
        base_digest=bundle["checkpoint_digest"],
        views=int(cfg.views),
        inference_seed=int(cfg.inference_seed),
        policies=policies,
    )
    (artifacts / "all_policies_before_test.json").write_text(
        json.dumps(frozen, indent=2)
    )
    return frozen


def summarize_test(cfg, bundle, root, device, frozen):
    root = Path(root)
    cache = root / "cache"
    artifacts = root / "artifacts"
    if frozen["base_digest"] != bundle["checkpoint_digest"]:
        raise ValueError("Frozen policy belongs to another base")
    probs = np.load(cache / "test_probs.npy", mmap_mode="r")
    features = np.load(cache / "test_features.npy", mmap_mode="r")
    labels = np.load(cache / "test_labels.npy")
    base = probs[:, 0]
    standard = probs.mean(1)
    standard_pred = standard.argmax(1)
    msp = 1 - base.max(1)
    rows = []

    def record(name, seed, probabilities, score):
        # Error detection always refers to identical original-view base errors.
        original = error_metrics(labels, base, score)
        final = error_metrics(labels, probabilities)
        interval = paired_accuracy_interval(
            labels, standard_pred, probabilities.argmax(1)
        )
        row = dict(
            method=name,
            seed=seed,
            acc=final["acc"],
            ece=final["ece"],
            base_error_auroc=original["auroc"],
            base_error_auprc=original["auprc"],
            gain_vs_tta_pp=interval["gain_pp"],
            ci_low_pp=interval["ci_low_pp"],
            ci_high_pp=interval["ci_high_pp"],
        )
        rows.append(row)
        np.savez_compressed(
            artifacts / f"test_{name}_{seed}.npz",
            ids=np.load(cache / "test_ids.npy"),
            labels=labels,
            probabilities=probabilities,
            base_badness=score,
        )

    record("base", "none", base, msp)
    record("tta", "none", standard, msp)
    for policy in frozen["policies"]:
        mode, seed = policy["mode"], policy["seed"]
        if mode == "msp":
            badness = 1 - probs.max(2)
        else:
            model = load_head(cfg, bundle, mode, seed, root, device)
            badness = trajectory_scores(model, features, probs, cfg, device)
        weighted = (
            standard
            if policy["temperature"] == "uniform"
            else weighted_consensus(probs, badness, policy["temperature"])
        )
        np.save(artifacts / f"test_badness_{policy['name']}.npy", badness)
        record("uw_" + mode, "none" if seed is None else seed, weighted, badness[:, 0])
    write_csv(root / "tables/comparison.csv", rows)
    aggregate = []
    for method in dict.fromkeys(r["method"] for r in rows):
        group = [r for r in rows if r["method"] == method]
        a = dict(
            method=method,
            n_meta_seeds=len(group) if method in {"uw_mean", "uw_full"} else 0,
        )
        for key in [
            "acc",
            "ece",
            "base_error_auroc",
            "base_error_auprc",
            "gain_vs_tta_pp",
        ]:
            a[key] = float(np.mean([r[key] for r in group]))
            a[key + "_sd"] = (
                float(np.std([r[key] for r in group], ddof=1))
                if len(group) > 1
                else 0.0
            )
        aggregate.append(a)
    write_csv(root / "tables/aggregate.csv", aggregate)
    return dict(
        status="complete",
        results=rows,
        aggregate=aggregate,
        test_samples=len(labels),
        views=int(cfg.views),
        new_base_digest=bundle["checkpoint_digest"],
    )
