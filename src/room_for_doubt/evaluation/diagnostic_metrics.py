"""Diagnostics with explicit targets; no tie-order-dependent AURC."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def detection_metrics(events, scores):
    events, scores = np.asarray(events), np.asarray(scores)
    if events.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Aligned finite detection scores required")
    valid = len(np.unique(events)) == 2
    return dict(
        auroc=float(roc_auc_score(events, scores)) if valid else None,
        auprc=float(average_precision_score(events, scores)) if valid else None,
        event_rate=float(events.mean()),
    )


def probability_metrics(labels, probabilities):
    labels, p = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or len(labels) != len(p) or not np.isfinite(p).all():
        raise ValueError("Aligned finite probabilities required")
    if np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Rows must be probability distributions")
    correct = p.argmax(1) == labels
    confidence = p.max(1)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    true = p[np.arange(len(p)), labels]
    return dict(
        acc=float(correct.mean()),
        ece=float(ece),
        nll=float(-np.log(np.clip(true, 1e-12, 1)).mean()),
        brier=float((np.square(p).sum(1) - 2 * true + 1).mean()),
    )


def regression_metrics(target, predicted):
    target = np.asarray(target).reshape(-1).astype(float)
    predicted = np.asarray(predicted).reshape(-1).astype(float)
    if target.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError("Aligned finite regression values required")
    mse = np.square(target - predicted).mean()
    variance = target.var()
    correlation = (
        float(np.corrcoef(target, predicted)[0, 1])
        if target.std() > 1e-12 and predicted.std() > 1e-12
        else None
    )
    return dict(
        mse=float(mse),
        r2=float(1 - mse / variance) if variance > 1e-12 else None,
        correlation=correlation,
        target_sd=float(target.std()),
        predicted_sd=float(predicted.std()),
    )


def class_average(values, labels, fit_rows, classes):
    """Class-only control fitted exclusively on the meta-training partition."""
    values, labels = np.asarray(values), np.asarray(labels)
    global_mean = values[fit_rows].mean(0)
    result = np.repeat(global_mean[None], classes, axis=0)
    for cls in range(classes):
        rows = fit_rows[labels[fit_rows] == cls]
        if len(rows):
            result[cls] = values[rows].mean(0)
    return result


def static_badness(probabilities, mode):
    p = np.asarray(probabilities)
    if mode == "msp":
        return 1 - p.max(-1)
    if mode == "entropy":
        return -(p * np.log(np.clip(p, 1e-12, 1))).sum(-1) / np.log(p.shape[-1])
    if mode == "gap":
        top = np.partition(p, -2, axis=-1)[..., -2:]
        return 1 - (top.max(-1) - top.min(-1))
    raise ValueError("Unknown static score")
