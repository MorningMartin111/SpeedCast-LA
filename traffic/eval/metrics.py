"""Masked error metrics for forecasts in miles per hour."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def masked_scores(
    y_true_mph: ArrayLike, y_pred_mph: ArrayLike, mask: ArrayLike
) -> dict[str, int | float | None]:
    """Score all observed readings together and return None metrics for an empty mask."""
    truth = np.asarray(y_true_mph)
    predictions = np.asarray(y_pred_mph)
    observed = np.asarray(mask)
    if truth.shape != predictions.shape or truth.shape != observed.shape:
        raise ValueError("Ground truth, predictions, and mask must have the same shape")
    if observed.dtype != np.bool_:
        raise ValueError("Mask must be Boolean")

    errors = np.subtract(predictions[observed], truth[observed], dtype=np.float64)
    n = errors.size
    if n == 0:
        return {"n": 0, "mae_mph": None, "rmse_mph": None, "bias_mph": None}
    if not np.isfinite(errors).all():
        raise ValueError("Observed readings must have finite prediction errors")

    return {
        "n": n,
        "mae_mph": float(np.abs(errors).mean()),
        "rmse_mph": float(np.sqrt(np.square(errors).mean())),
        "bias_mph": float(errors.mean()),
    }
