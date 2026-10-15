"""Validation-only selection and metric definitions for the fixed campaign."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from room_for_doubt.evaluation.metrics import calculate_aurc


def top2_correct(top, badness, threshold):
    mask = (badness[:, 0] > threshold) & (badness[:, 1] < threshold)
    return np.where(mask, top[:, 1], top[:, 0]), mask


def select_threshold(top, badness, noisy_labels, thresholds):
    rows = []
    for t in thresholds:
        pred, mask = top2_correct(top, badness, t)
        rows.append(
            dict(
                threshold=float(t),
                val_noisy_acc=float(np.mean(pred == noisy_labels)),
                overrides=int(mask.sum()),
            )
        )
    # No-op threshold wins ties, then larger threshold: fixed conservative rule.
    best = max(rows, key=lambda r: (r["val_noisy_acc"], r["threshold"]))
    return best["threshold"], rows


def weighted_consensus(probs, badness, temperature):
    logw = -badness / temperature
    logw -= logw.max(axis=1, keepdims=True)
    weights = np.exp(logw)
    weights /= weights.sum(axis=1, keepdims=True)
    return (probs * weights[:, :, None]).sum(axis=1)


def select_temperature(probs, badness, noisy_labels, temperatures):
    rows = []
    # Equal weighting is an exact no-op option, not a finite approximation.
    standard = probs.mean(axis=1)
    rows.append(
        dict(
            temperature="uniform",
            val_noisy_acc=float(np.mean(standard.argmax(1) == noisy_labels)),
        )
    )
    for t in temperatures:
        pred = weighted_consensus(probs, badness, t).argmax(1)
        rows.append(
            dict(
                temperature=float(t), val_noisy_acc=float(np.mean(pred == noisy_labels))
            )
        )
    best = max(range(len(rows)), key=lambda i: rows[i]["val_noisy_acc"])
    return rows[best]["temperature"], rows


def error_metrics(labels, probabilities, badness=None):
    pred = probabilities.argmax(1)
    errors = (pred != labels).astype(int)
    confidence = probabilities.max(1)
    score = 1 - confidence if badness is None else badness
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(
                (1 - errors[mask]).mean() - confidence[mask].mean()
            )
    return dict(
        acc=float((1 - errors).mean()),
        auroc=float(roc_auc_score(errors, score)),
        auprc=float(average_precision_score(errors, score)),
        aurc=float(calculate_aurc(-score, errors)),
        ece=float(ece),
    )


def paired_accuracy_interval(labels, baseline, proposed, seed=20261015):
    """Paired test-example bootstrap; conditional on the fitted models/split."""
    delta = (np.asarray(proposed) == labels).astype(int) - (
        np.asarray(baseline) == labels
    ).astype(int)
    counts = np.array([(delta == v).sum() for v in [-1, 0, 1]])
    draws = np.random.default_rng(seed).multinomial(
        len(delta), counts / len(delta), size=2000
    )
    gains = (draws[:, 2] - draws[:, 0]) / len(delta) * 100
    low, high = np.quantile(gains, [0.025, 0.975])
    return dict(
        gain_pp=float(delta.mean() * 100), ci_low_pp=float(low), ci_high_pp=float(high)
    )
