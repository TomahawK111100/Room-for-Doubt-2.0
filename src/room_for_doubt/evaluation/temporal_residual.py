"""Noisy-validation policy selection and explicitly targeted error metrics."""

import numpy as np
import torch
from room_for_doubt.models.temporal_residual import residual_posterior
from room_for_doubt.evaluation.diagnostic_metrics import (
    probability_metrics,
    detection_metrics,
)


def corrected_probabilities(base, residual, alpha):
    p = torch.as_tensor(np.array(base), dtype=torch.float32)
    h = torch.as_tensor(np.array(residual), dtype=torch.float32)
    return residual_posterior(p, h, alpha).numpy()


def select_alpha(base, residual, noisy_labels, alphas):
    if (
        not alphas
        or alphas != sorted(set(alphas))
        or alphas[0] != 0
        or any(not 0 <= a <= 1 for a in alphas)
    ):
        raise ValueError("Ordered alpha grid in [0,1] must include exact identity")
    rows = []
    for alpha in alphas:
        metrics = probability_metrics(
            noisy_labels, corrected_probabilities(base, residual, alpha)
        )
        rows.append(
            dict(
                alpha=float(alpha),
                val_noisy_acc=metrics["acc"],
                val_noisy_nll=metrics["nll"],
            )
        )
    selected = max(
        rows, key=lambda r: (r["val_noisy_acc"], -r["val_noisy_nll"], -r["alpha"])
    )
    return selected, rows


def evaluate_probabilities(labels, base, corrected):
    """Separate deployed-model errors from the fixed original-base error event."""
    rows = {}
    for name, p in (("identity", base), ("temporal_residual", corrected)):
        metrics = probability_metrics(labels, p)
        score = 1 - p.max(1)
        own = detection_metrics(p.argmax(1) != labels, score)
        fixed = detection_metrics(base.argmax(1) != labels, score)
        rows[name] = dict(
            **metrics,
            own_error_auroc=own["auroc"],
            own_error_auprc=own["auprc"],
            original_base_error_auroc=fixed["auroc"],
        )
    return rows
