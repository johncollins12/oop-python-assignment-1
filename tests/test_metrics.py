"""Tests for a1.metrics: hand-checked values, properties and invalid inputs."""

import math

import numpy as np
import pytest

from a1.metrics import evaluate, mae, mape, rmse

ACTUAL = [100, 200, 300]
PREDICTED = [110, 190, 330]


def test_values_match_hand_calculation():
    # |errors| = 10, 10, 30 ; squared = 100, 100, 900 ; percentages = 10%, 5%, 10%
    assert mae(ACTUAL, PREDICTED) == pytest.approx(50 / 3)
    assert rmse(ACTUAL, PREDICTED) == pytest.approx(math.sqrt(1100 / 3))
    assert mape(ACTUAL, PREDICTED) == pytest.approx(25 / 3)


def test_perfect_forecast_gives_zero_error():
    for metric in (mae, rmse, mape):
        assert metric(ACTUAL, ACTUAL) == 0.0


def test_rmse_is_never_below_mae():
    rng = np.random.default_rng(42)
    actual = rng.uniform(100, 1000, size=50)
    predicted = actual + rng.normal(0, 25, size=50)
    assert rmse(actual, predicted) >= mae(actual, predicted)


def test_evaluate_returns_all_three_metrics():
    result = evaluate(ACTUAL, PREDICTED)
    assert set(result) == {"MAE", "RMSE", "MAPE"}
    assert result["MAE"] == pytest.approx(mae(ACTUAL, PREDICTED))


@pytest.mark.parametrize(
    "actual, predicted",
    [
        ([], []),                      # empty input
        ([1, 2, 3], [1, 2]),           # unequal lengths
        ([1, float("nan"), 3], [1, 2, 3]),  # NaN
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),  # not one-dimensional
    ],
)
def test_invalid_inputs_are_rejected(actual, predicted):
    for metric in (mae, rmse, mape):
        with pytest.raises(ValueError):
            metric(actual, predicted)


def test_mape_rejects_zero_actual():
    with pytest.raises(ValueError, match="zero"):
        mape([0, 100], [5, 100])