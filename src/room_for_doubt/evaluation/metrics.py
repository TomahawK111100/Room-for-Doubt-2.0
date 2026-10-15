"""Uncertainty evaluation metrics; high confidence means likely correct."""

import numpy as np


def calculate_aurc(confidences, errors):
    """
    Calculate Area Under the Risk-Coverage (AURC) curve.
    confidences: array-like of confidence scores (higher is more confident)
    errors: array-like of binary errors (1 if error, 0 if correct)
    """
    confidences = np.array(confidences)
    errors = np.array(errors)

    # Sort descending by confidence
    sorted_indices = np.argsort(confidences)[::-1]
    sorted_errors = errors[sorted_indices]

    # Calculate cumulative error rate (risk)
    coverages = np.arange(1, len(sorted_errors) + 1)
    risks = np.cumsum(sorted_errors) / coverages

    # AURC is the mean of risks
    return np.mean(risks)
