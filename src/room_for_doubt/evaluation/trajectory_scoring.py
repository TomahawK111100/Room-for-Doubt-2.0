"""Class-count independent scores and validation-only threshold candidates."""

import numpy as np


def mean_margin_badness(trajectories):
    """Low predicted probability margin means high badness; not calibrated P(error).

    Takes the mean predicted trajectory (for stochastic models, average draws
    before calling). Fixed theoretical margin range [-1,1] defines the scale.
    No clean labels, fitted detector, or model-output maximum is used.
    """
    values = np.asarray(trajectories)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite matrix of predicted trajectories")
    return np.clip((1 - values.mean(axis=1)) / 2, 0, 1)


def quantile_thresholds(validation_badness, quantiles):
    values = np.asarray(validation_badness)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Expected paired Top-1/Top-2 validation scores")
    # Candidate thresholds depend on score scale, not class count or clean labels.
    return np.unique(np.r_[1.0, np.quantile(values.reshape(-1), quantiles)]).tolist()
