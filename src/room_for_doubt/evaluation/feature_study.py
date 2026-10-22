"""Matched encoder/target diagnostics and validation-only joint policy choice."""

import json
from pathlib import Path
import numpy as np
from room_for_doubt.evaluation.corrected_campaign import predict_arrays
from room_for_doubt.evaluation.diagnostic_metrics import (
    regression_metrics,
    probability_metrics,
    detection_metrics,
    static_badness,
)
from room_for_doubt.evaluation.selection import (
    select_temperature,
    weighted_consensus,
    paired_accuracy_interval,
)
from room_for_doubt.evaluation.trajectory_scoring import mean_margin_badness
from room_for_doubt.training.feature_heads import load_feature_head, target_values
from room_for_doubt.reporting.artifacts import write_csv


def head_specs(cfg):
    return [
        (encoder, mode, int(seed))
        for encoder in cfg.encoders
        for mode in cfg.target_modes
        for seed in cfg.seeds
    ]


def name_of(spec):
    return f"{spec[0]}__{spec[1]}__{spec[2]}"


def score_heads(split, cfg, out, probabilities, projected, device):
    scores = {
        name: static_badness(probabilities, name) for name in ("msp", "gap", "entropy")
    }
    for spec in head_specs(cfg):
        encoder, mode, seed = spec
        name = name_of(spec)
        path = Path(out) / "artifacts" / f"{split}_{name}.npy"
        if path.exists():
            score = np.load(path)
        else:
            model = load_feature_head(cfg, out, encoder, mode, seed, device)
            f = projected[encoder]
            paths = predict_arrays(
                model,
                f.reshape(-1, cfg.feature_dim),
                probabilities.argmax(-1).reshape(-1),
                cfg,
                device,
            )
            score = mean_margin_badness(paths).reshape(probabilities.shape[:2])
            np.save(path, score)
            del model
        if score.shape != probabilities.shape[:2] or not np.isfinite(score).all():
            raise ValueError("Invalid feature-study view scores")
        scores[name] = score
    return scores


def freeze_policies(cfg, out, probabilities, labels, projected, device):
    scores = score_heads("val", cfg, out, probabilities, projected, device)
    policies, sweeps = [], []
    for count in cfg.view_counts:
        p = probabilities[:, :count]
        policies.append(
            dict(
                method="tta",
                views=int(count),
                temperature="uniform",
                val_noisy_acc=float((p.mean(1).argmax(1) == labels).mean()),
            )
        )
        for name, score in scores.items():
            temperature, rows = select_temperature(
                p, score[:, :count], labels, cfg.temperatures
            )
            chosen = next(r for r in rows if r["temperature"] == temperature)
            policies.append(
                dict(
                    method=name,
                    views=int(count),
                    temperature=temperature,
                    val_noisy_acc=chosen["val_noisy_acc"],
                )
            )
            sweeps.extend(dict(method=name, views=int(count), **r) for r in rows)
    # Fixed order gives smaller view budget priority in an accuracy tie. Within
    # each budget, uniform weighting already wins temperature ties.
    selected = {}
    for policy in policies:
        name = policy["method"]
        if (
            name not in selected
            or policy["val_noisy_acc"] > selected[name]["val_noisy_acc"]
        ):
            selected[name] = policy
    frozen = dict(
        base_digest=cfg.base_digest,
        policies=policies,
        selected=selected,
        selection="maximize noisy-validation accuracy; ties prefer fewer views then uniform",
    )
    path = Path(out) / "artifacts/policies_before_test.json"
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError("Previously frozen feature policies differ")
    path.write_text(json.dumps(frozen, indent=2))
    write_csv(Path(out) / "tables/validation_sweeps.csv", sweeps)
    write_csv(Path(out) / "tables/validation_selected.csv", list(selected.values()))
    return frozen


def weighted(probabilities, scores, policy):
    p = probabilities[:, : policy["views"]]
    if policy["temperature"] == "uniform":
        return p.mean(1)
    return weighted_consensus(
        p, scores[policy["method"]][:, : policy["views"]], policy["temperature"]
    )


def evaluate_policies(cfg, out, probabilities, labels, projected, device, frozen):
    if (
        json.loads((Path(out) / "artifacts/policies_before_test.json").read_text())
        != frozen
    ):
        raise ValueError("All policies must be frozen before final evaluation")
    scores = score_heads("test", cfg, out, probabilities, projected, device)
    base = probabilities[:, 0]
    selected_tta = weighted(probabilities, scores, frozen["selected"]["tta"])
    bootstrap_seed = int(cfg.get("bootstrap_seed", 20261015))
    errors = base.argmax(1) != labels
    rows = []
    for policy in [dict(method="base", views=1, temperature="uniform")] + frozen[
        "policies"
    ]:
        name, count = policy["method"], policy["views"]
        p = base if name == "base" else weighted(probabilities, scores, policy)
        score = scores[name][:, 0] if name in scores else scores["msp"][:, 0]
        row = dict(
            method=name,
            views=count,
            temperature=policy["temperature"],
            tta_reference_views=count,
            selected_tta_reference_views=frozen["selected"]["tta"]["views"],
            selected_by_validation=name == "base" or frozen["selected"][name] == policy,
            **probability_metrics(labels, p),
        )
        row.update(
            {"base_error_" + k: v for k, v in detection_metrics(errors, score).items()}
        )
        msp_policy = next(
            x
            for x in frozen["policies"]
            if x["method"] == "msp" and x["views"] == count
        )
        for reference, q in (
            ("base", base),
            ("tta", probabilities[:, :count].mean(1)),
            ("tta_selected", selected_tta),
            ("msp", weighted(probabilities, scores, msp_policy)),
        ):
            row.update(
                {
                    f"{k}_vs_{reference}": v
                    for k, v in paired_accuracy_interval(
                        labels, q.argmax(1), p.argmax(1), seed=bootstrap_seed
                    ).items()
                }
            )
        rows.append(row)
    write_csv(Path(out) / "tables/test_all_budgets.csv", rows)
    selected = [r for r in rows if r["selected_by_validation"]]
    write_csv(Path(out) / "tables/test_validation_selected.csv", selected)
    groups = {}
    for r in selected:
        family = r["method"].rsplit("__", 1)[0] if "__" in r["method"] else r["method"]
        groups.setdefault(family, []).append(r)
    aggregate = []
    for family, group in groups.items():
        aggregate.append(
            dict(
                method=family,
                n_meta_seeds=len(group) if "__" in family else 0,
                acc=float(np.mean([r["acc"] for r in group])),
                acc_sd=float(np.std([r["acc"] for r in group], ddof=1))
                if len(group) > 1
                else 0.0,
                error_auroc=float(np.mean([r["base_error_auroc"] for r in group])),
                gain_vs_base_pp=float(np.mean([r["gain_pp_vs_base"] for r in group])),
                gain_vs_selected_tta_pp=float(
                    np.mean([r["gain_pp_vs_tta_selected"] for r in group])
                ),
                selected_views="/".join(str(r["views"]) for r in group),
            )
        )
    write_csv(Path(out) / "tables/aggregate.csv", aggregate)
    return selected, aggregate


def reconstruction(cfg, bundle, out, projected, best_margin, device):
    data = bundle["data"]
    lookup = {int(s): i for i, s in enumerate(data["sample_ids"])}
    rows = []
    for spec in head_specs(cfg):
        encoder, mode, seed = spec
        model = load_feature_head(cfg, out, encoder, mode, seed, device)
        paths = predict_arrays(
            model, projected[encoder], data["noisy_labels"], cfg, device
        )
        np.save(
            Path(out) / "artifacts" / f"train_badness_{name_of(spec)}.npy",
            mean_margin_badness(paths),
        )
        target = target_values(data, best_margin, mode, cfg.early_epochs)[:, 0]
        for partition in ("meta_train", "meta_val", "detector_eval"):
            indices = np.array([lookup[int(s)] for s in data[partition + "_ids"]])
            rows.append(
                dict(
                    encoder=encoder,
                    target=mode,
                    seed=seed,
                    partition=partition,
                    samples=len(indices),
                    **regression_metrics(target[indices], paths[indices, 0]),
                )
            )
        del model
    write_csv(Path(out) / "tables/reconstruction.csv", rows)


def noise_and_epoch_profile(cfg, bundle, out, clean):
    """Final evaluation only: real-history signal versus its distilled versions."""
    data = bundle["data"]
    lookup = {int(s): i for i, s in enumerate(data["sample_ids"])}
    indices = np.array([lookup[int(s)] for s in data["detector_eval_ids"]])
    events = (
        data["noisy_labels"][indices] != np.asarray(clean)[data["sample_ids"][indices]]
    )
    trajectories = data["targets"][indices]
    rows = [
        dict(
            epoch=int(epoch),
            samples=len(indices),
            **detection_metrics(events, (1 - trajectories[:, epoch]) / 2),
        )
        for epoch in range(trajectories.shape[1])
    ]
    write_csv(Path(out) / "tables/real_history_epoch_noise.csv", rows)
    window_rows = []
    for name, values in (
        ("real_early_mean", trajectories[:, : cfg.early_epochs].mean(1)),
        ("real_history_mean", trajectories.mean(1)),
        ("real_last_five", trajectories[:, -5:].mean(1)),
    ):
        window_rows.append(
            dict(
                method=name,
                samples=len(indices),
                **detection_metrics(events, (1 - values) / 2),
            )
        )
    for spec in head_specs(cfg):
        path = Path(out) / "artifacts" / f"train_badness_{name_of(spec)}.npy"
        window_rows.append(
            dict(
                method=name_of(spec),
                samples=len(indices),
                **detection_metrics(events, np.load(path)[indices]),
            )
        )
    write_csv(Path(out) / "tables/train_noise_diagnostic.csv", window_rows)
