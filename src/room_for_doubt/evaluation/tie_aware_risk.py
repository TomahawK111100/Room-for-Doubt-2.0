"""Risk-coverage area with expected random ordering within exact confidence ties."""

import numpy as np


def expected_tie_aurc(confidence, errors):
    confidence, errors = np.asarray(confidence), np.asarray(errors)
    if confidence.ndim != 1 or confidence.shape != errors.shape or len(errors) == 0:
        raise ValueError("Aligned nonempty confidence and error vectors required")
    if not np.isfinite(confidence).all() or not np.isin(errors, [0, 1]).all():
        raise ValueError("Finite confidence and binary errors required")
    order = np.argsort(-confidence, kind="stable")
    c, e = confidence[order], errors[order].astype(float)
    starts = np.r_[0, np.flatnonzero(c[1:] != c[:-1]) + 1]
    total, previous_errors = 0.0, 0.0
    for start, stop in zip(starts, np.r_[starts[1:], len(e)]):
        size = stop - start
        count = float(e[start:stop].sum())
        j = np.arange(1, size + 1, dtype=float)
        total += np.sum((previous_errors + j * count / size) / (start + j))
        previous_errors += count
    return float(total / len(e))
