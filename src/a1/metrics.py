"""Forecast-accuracy metrics shared by the mini-projects.

Each function compares a 1-D array of actual values with a 1-D array of
predictions of the same length and returns a single float.
"""

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["mae", "rmse", "mape", "evaluate"]


def _validate_pair(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert both inputs to 1-D float arrays and check they are comparable.

    Raises
    ------
    ValueError
        If an input is not one-dimensional, is empty, the lengths differ,
        or any value is NaN or infinite.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if a.ndim != 1 or p.ndim != 1:
        raise ValueError("actual and predicted must be one-dimensional")
    if a.size == 0 or p.size == 0:
        raise ValueError("actual and predicted must not be empty")
    if a.shape != p.shape:
        raise ValueError(f"length mismatch: {a.size} actual vs {p.size} predicted")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(p))):
        raise ValueError("actual and predicted must not contain NaN or infinite values")
    return a, p


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute error, in the same units as the data."""
    a, p = _validate_pair(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root mean squared error, in the same units as the data.

    Squaring before averaging penalises large errors more heavily than MAE,
    so RMSE >= MAE always holds.
    """
    a, p = _validate_pair(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean absolute percentage error, returned as a percentage (e.g. 8.3, not 0.083).

    Raises
    ------
    ValueError
        If any actual value is zero, because the percentage error is undefined.
    """
    a, p = _validate_pair(actual, predicted)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs((a - p) / np.abs(a))) * 100.0)


def evaluate(actual: ArrayLike, predicted: ArrayLike) -> dict[str, float]:
    """Return MAE, RMSE and MAPE together, as used in the comparison tables."""
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
    }