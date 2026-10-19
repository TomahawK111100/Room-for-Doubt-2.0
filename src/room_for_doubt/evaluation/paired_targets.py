"""Conditional paired comparisons; no fitting or policy selection."""

import numpy as np
from sklearn.metrics import roc_auc_score
from room_for_doubt.evaluation.selection import paired_accuracy_interval
from room_for_doubt.evaluation.feature_study import weighted


def paired_auroc_interval(events, reference, proposed, draws=2000, seed=20261019):
    """Stratified paired bootstrap with shared multiplicities and exact ties.

    Sorting is done once per score. Bootstrap class counts remain fixed; these
    intervals describe test sampling conditional on one fitted base and policy.
    """
    events = np.asarray(events)
    scores = [np.asarray(reference), np.asarray(proposed)]
    if (
        events.ndim != 1
        or not len(events)
        or not np.isin(events, [0, 1]).all()
        or any(s.shape != events.shape or not np.isfinite(s).all() for s in scores)
        or draws < 2
    ):
        raise ValueError("Require aligned binary events, finite scores and draws >= 2")
    pos, neg = np.flatnonzero(events == 1), np.flatnonzero(events == 0)
    if not len(pos) or not len(neg):
        raise ValueError("AUROC requires both error and correct examples")
    ranks = []
    for score in scores:
        order = np.argsort(score, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(score[order])) + 1]
        ranks.append((order, starts))

    def auc(counts, rank):
        order, starts = rank
        p = np.add.reduceat(counts[:, order] * events[order], starts, axis=1)
        n = np.add.reduceat(counts[:, order] * (1 - events[order]), starts, axis=1)
        return (p * (n.cumsum(1) - n / 2)).sum(1) / (len(pos) * len(neg))

    rng, values = np.random.default_rng(seed), []
    for start in range(0, draws, 16):
        count = min(16, draws - start)
        weights = np.zeros((count, len(events)), dtype=np.int32)
        weights[:, pos] = rng.multinomial(
            len(pos), np.full(len(pos), 1 / len(pos)), count
        )
        weights[:, neg] = rng.multinomial(
            len(neg), np.full(len(neg), 1 / len(neg)), count
        )
        values.extend(auc(weights, ranks[1]) - auc(weights, ranks[0]))
    low, high = np.quantile(values, [0.025, 0.975])
    return dict(
        auroc_difference=float(
            roc_auc_score(events, scores[1]) - roc_auc_score(events, scores[0])
        ),
        auroc_ci_low=float(low),
        auroc_ci_high=float(high),
    )


def target_contrasts(
    cfg, out, probabilities, labels, ids, frozen, draws=2000, bootstrap_seed=20261019
):
    """Compare Early-5 against each declared static/history target per head.

    Each method keeps its own saved validation-selected policy. AUROC refers to
    the same original-view base-error event, independently of selected TTA views.
    """
    from pathlib import Path
    import json
    from room_for_doubt.reporting.artifacts import write_csv

    out = Path(out)
    if (
        json.loads((out / "artifacts/policies_before_test.json").read_text()) != frozen
        or frozen["base_digest"] != cfg.base_digest
    ):
        raise ValueError("Require the saved frozen source policies")
    labels, ids = np.asarray(labels), np.asarray(ids)
    if (
        probabilities.ndim != 3
        or labels.shape != (len(probabilities),)
        or ids.shape != labels.shape
        or len(np.unique(ids)) != len(ids)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(-1), 1, atol=1e-5)
        or not np.issubdtype(labels.dtype, np.integer)
        or np.any(labels < 0)
        or np.any(labels >= probabilities.shape[-1])
    ):
        raise ValueError("Invalid paired prediction identities or probabilities")
    scores, predictions = {}, {}
    for name, policy in frozen["selected"].items():
        if "__" not in name:
            continue
        score = np.load(out / "artifacts" / f"test_{name}.npy")
        if score.shape != probabilities.shape[:2] or not np.isfinite(score).all():
            raise ValueError("Target score dimensions or values differ")
        if (
            policy["method"] != name
            or policy["views"] not in cfg.view_counts
            or (
                policy["temperature"] != "uniform"
                and policy["temperature"] not in cfg.temperatures
            )
        ):
            raise ValueError("Selected policy is outside the saved search")
        scores[name] = score
        predictions[name] = weighted(probabilities, {name: score}, policy).argmax(1)
    rows = []
    events = probabilities[:, 0].argmax(1) != labels
    for encoder in cfg.encoders:
        for seed in cfg.seeds:
            proposed = f"{encoder}__early_mean__{seed}"
            for mode in ("best_margin", "history_mean", "best_augmented_mean"):
                if mode not in cfg.target_modes:
                    continue
                reference = f"{encoder}__{mode}__{seed}"
                if proposed not in scores or reference not in scores:
                    raise ValueError("Missing a declared paired target")
                rows.append(
                    dict(
                        encoder=encoder,
                        head_seed=int(seed),
                        reference=mode,
                        proposed="early_mean",
                        comparison="own_validation_selected_policy",
                        samples=len(ids),
                        error_event="original_view_base_error",
                        reference_views=frozen["selected"][reference]["views"],
                        proposed_views=frozen["selected"][proposed]["views"],
                        **paired_accuracy_interval(
                            labels,
                            predictions[reference],
                            predictions[proposed],
                            bootstrap_seed,
                        ),
                        **paired_auroc_interval(
                            events,
                            scores[reference][:, 0],
                            scores[proposed][:, 0],
                            draws,
                            bootstrap_seed,
                        ),
                    )
                )
    if not rows:
        raise ValueError("No declared target comparisons")
    write_csv(out / "tables/paired_target_contrasts.csv", rows)
    # IDs and predictions make the compact contrast table independently auditable.
    np.savez_compressed(
        out / "artifacts/paired_selected_predictions.npz",
        ids=ids,
        labels=labels,
        **predictions,
    )
    return rows
