"""Frozen validation policies, matched controls and final exploratory evaluation."""

import json
from pathlib import Path
import numpy as np
from room_for_doubt.evaluation.corrected_campaign import predict_arrays
from room_for_doubt.evaluation.trajectory_scoring import mean_margin_badness
from room_for_doubt.evaluation.selection import (
    select_temperature,
    weighted_consensus,
    paired_accuracy_interval,
)
from room_for_doubt.evaluation.diagnostic_metrics import (
    class_average,
    regression_metrics,
    probability_metrics,
    detection_metrics,
    static_badness,
)
from room_for_doubt.training.post_kaggle_meta import load_head
from room_for_doubt.training.diagnostic_controls import load_control, control_target
from room_for_doubt.reporting.artifacts import write_csv


def method_specs(cfg, post_cfg):
    return [
        (mode, int(seed), "existing")
        for mode in ("mean", "full")
        for seed in post_cfg.seeds
    ] + [
        (mode, int(seed), "control") for mode in cfg.control_modes for seed in cfg.seeds
    ]


def get_model(spec, cfg, bundle, post_cfg, post, out, device):
    mode, seed, source = spec
    if source == "existing":
        return load_head(post_cfg, bundle, mode, seed, post, device)
    return load_control(Path(out) / "heads" / f"{mode}_{seed}", cfg, mode, seed, device)


def reconstruction(cfg, bundle, post_cfg, post, out, features, best_margin, device):
    data = bundle["data"]
    lookup = {int(s): i for i, s in enumerate(data["sample_ids"])}
    partitions = {
        part: np.array([lookup[int(s)] for s in data[part + "_ids"]])
        for part in ("meta_train", "meta_val", "detector_eval")
    }
    fit = partitions["meta_train"]
    rows = []
    for spec in method_specs(cfg, post_cfg):
        mode, seed, _ = spec
        model = get_model(spec, cfg, bundle, post_cfg, post, out, device)
        predicted = predict_arrays(model, features, data["noisy_labels"], cfg, device)
        target = (
            data["targets"]
            if mode == "full"
            else (
                np.repeat(data["targets"].mean(1, keepdims=True), 50, axis=1)
                if mode == "mean"
                else control_target(data, best_margin, mode, cfg.early_epochs)
            )
        )
        class_values = class_average(
            target, data["noisy_labels"], fit, cfg.dataset.num_classes
        )
        global_values = np.repeat(
            target[fit].mean(0, keepdims=True), len(target), axis=0
        )
        np.save(
            Path(out) / "artifacts" / f"observed_train_badness_{mode}_{seed}.npy",
            mean_margin_badness(predicted),
        )
        for part, indices in partitions.items():
            for estimator, values in (
                ("head", predicted),
                ("class_only", class_values[data["noisy_labels"]]),
                ("constant", global_values),
            ):
                result = regression_metrics(target[indices], values[indices])
                rows.append(
                    dict(
                        mode=mode,
                        seed=seed,
                        partition=part,
                        estimator=estimator,
                        samples=len(indices),
                        **result,
                    )
                )
        del model
    write_csv(Path(out) / "tables/reconstruction.csv", rows)
    return rows


def collect_scores(split, cfg, bundle, post_cfg, post, out, class_values, device):
    cache = Path(post) / "cache"
    probs = np.load(cache / f"{split}_probs.npy", mmap_mode="r")
    features = np.load(cache / f"{split}_features.npy", mmap_mode="r")
    scores = {mode: static_badness(probs, mode) for mode in ("msp", "entropy", "gap")}
    scores["class_history"] = np.clip((1 - class_values[probs.argmax(-1)]) / 2, 0, 1)
    for spec in method_specs(cfg, post_cfg):
        mode, seed, source = spec
        name = f"{mode}_{seed}"
        path = Path(out) / "artifacts" / f"{split}_badness_{name}.npy"
        if source == "existing":
            score = np.load(Path(post) / "artifacts" / f"{split}_badness_{name}.npy")
        elif path.exists():
            score = np.load(path)
        else:
            model = get_model(spec, cfg, bundle, post_cfg, post, out, device)
            paths = predict_arrays(
                model,
                features.reshape(-1, 512),
                probs.argmax(-1).reshape(-1),
                cfg,
                device,
            )
            score = mean_margin_badness(paths).reshape(probs.shape[:2])
            np.save(path, score)
            del model
        if score.shape != probs.shape[:2] or not np.isfinite(score).all():
            raise ValueError("Invalid view scores: " + name)
        scores[name] = score
    return probs, scores


def freeze_diagnostic_policies(cfg, bundle, post_cfg, post, out, class_values, device):
    probs, scores = collect_scores(
        "val", cfg, bundle, post_cfg, post, out, class_values, device
    )
    labels = np.load(Path(post) / "cache/val_labels.npy")
    rows, sweeps, policies = [], [], []
    for count in cfg.view_counts:
        for name, score in scores.items():
            temperature, sweep = select_temperature(
                probs[:, :count], score[:, :count], labels, cfg.temperatures
            )
            policies.append(
                dict(method=name, views=int(count), temperature=temperature)
            )
            chosen = next(r for r in sweep if r["temperature"] == temperature)
            rows.append(
                dict(
                    method=name,
                    views=int(count),
                    temperature=temperature,
                    val_noisy_acc=chosen["val_noisy_acc"],
                    gain_vs_uniform_pp=100
                    * (chosen["val_noisy_acc"] - sweep[0]["val_noisy_acc"]),
                )
            )
            sweeps.extend(dict(method=name, views=int(count), **r) for r in sweep)
    frozen = dict(
        base_digest=bundle["checkpoint_digest"],
        policies=policies,
        scope="Exploratory fixed diagnostic battery; no test-driven selection.",
        source_post_run=str(post),
    )
    path = Path(out) / "artifacts/diagnostic_policies_before_test.json"
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError("Previously frozen diagnostic policies differ")
    path.write_text(json.dumps(frozen, indent=2))
    write_csv(Path(out) / "tables/validation_policies.csv", rows)
    write_csv(Path(out) / "tables/validation_sweeps.csv", sweeps)
    return frozen


def apply_policy(probs, score, policy):
    p, s = probs[:, : policy["views"]], score[:, : policy["views"]]
    return (
        p.mean(1)
        if policy["temperature"] == "uniform"
        else weighted_consensus(p, s, policy["temperature"])
    )


def evaluate_diagnostic_policies(
    cfg, bundle, post_cfg, post, out, class_values, device, frozen
):
    marker = Path(out) / "artifacts/diagnostic_policies_before_test.json"
    if json.loads(marker.read_text()) != frozen:
        raise ValueError("Freeze all policies before test evaluation")
    probs, scores = collect_scores(
        "test", cfg, bundle, post_cfg, post, out, class_values, device
    )
    labels = np.load(Path(post) / "cache/test_labels.npy")
    base = probs[:, 0]
    base_pred = base.argmax(1)
    errors = base_pred != labels
    rows = []
    for count in cfg.view_counts:
        policies = [p for p in frozen["policies"] if p["views"] == count]
        standard = probs[:, :count].mean(1)
        msp_policy = next(p for p in policies if p["method"] == "msp")
        msp_pred = apply_policy(probs, scores["msp"], msp_policy).argmax(1)
        entries = [
            ("base", base, scores["msp"][:, 0], "none"),
            ("tta", standard, scores["msp"][:, 0], "uniform"),
        ]
        entries += [
            (
                p["method"],
                apply_policy(probs, scores[p["method"]], p),
                scores[p["method"]][:, 0],
                p["temperature"],
            )
            for p in policies
        ]
        for name, weighted, badness, temperature in entries:
            pred = weighted.argmax(1)
            row = dict(
                method=name,
                views=int(count),
                temperature=temperature,
                **probability_metrics(labels, weighted),
            )
            row.update(
                {
                    "base_error_" + k: v
                    for k, v in detection_metrics(errors, badness).items()
                }
            )
            for reference, prediction in (
                ("base", base_pred),
                ("tta", standard.argmax(1)),
                ("msp", msp_pred),
            ):
                interval = paired_accuracy_interval(labels, prediction, pred)
                row.update({f"{k}_vs_{reference}": v for k, v in interval.items()})
            rows.append(row)
    write_csv(Path(out) / "tables/test_comparison.csv", rows)
    return rows


def noise_diagnostics(cfg, bundle, clean_labels, best_probabilities, out):
    """Clean train labels are accessed ONLY for this final diagnostic evaluation."""
    data = bundle["data"]
    lookup = {int(s): i for i, s in enumerate(data["sample_ids"])}
    indices = np.array([lookup[int(s)] for s in data["detector_eval_ids"]])
    events = (
        data["noisy_labels"][indices]
        != np.asarray(clean_labels)[data["sample_ids"][indices]]
    )
    sources = {
        "real_history_mean": mean_margin_badness(data["targets"])[indices],
        "best_observed_probability": 1
        - best_probabilities[indices, data["noisy_labels"][indices]],
    }
    for path in (Path(out) / "artifacts").glob("observed_train_badness_*.npy"):
        sources[path.stem.removeprefix("observed_train_badness_")] = np.load(path)[
            indices
        ]
    rows = [
        dict(
            method=name,
            samples=len(indices),
            evaluation="train_noise_meta_heldout_base_seen",
            **detection_metrics(events, score),
        )
        for name, score in sources.items()
    ]
    write_csv(Path(out) / "tables/train_noise_diagnostic.csv", rows)
    return rows
