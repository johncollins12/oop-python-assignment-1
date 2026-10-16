"""Tests for a1.forecasting: exact recovery, the interface contract and the bootstrap."""

import math

import numpy as np
import pytest

from a1.forecasting import (
    CAGRForecaster,
    FibonacciRatioForecaster,
    Forecaster,
    LinearTrendForecaster,
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
    SESForecaster,
    bootstrap_intervals,
    fibonacci_numbers,
    fibonacci_ratios,
    walk_forward,
)

ALL_MODELS = [
    LinearTrendForecaster,
    CAGRForecaster,
    FibonacciRatioForecaster,
    MovingAverageForecaster,
    SESForecaster,
]


# ----- the abstract base -------------------------------------------------------
def test_base_class_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Forecaster()


# ----- exact recovery of the generating model -----------------------------------
def test_linear_recovers_exact_line():
    y = 5 + 2 * np.arange(6)                      # 5, 7, 9, 11, 13, 15
    model = LinearTrendForecaster().fit(y)
    assert model.slope == pytest.approx(2.0)
    assert model.intercept == pytest.approx(5.0)
    np.testing.assert_allclose(model.predict(2), [17, 19])
    np.testing.assert_allclose(model.residuals(), 0, atol=1e-9)


def test_cagr_recovers_exact_geometric_series():
    y = 100 * 1.05 ** np.arange(7)
    model = CAGRForecaster().fit(y)
    assert model.rate == pytest.approx(0.05)
    np.testing.assert_allclose(model.predict(3), 100 * 1.05 ** np.arange(7, 10))
    np.testing.assert_allclose(model.fitted_values(), y)


def test_cagr_rejects_non_positive_values_and_stays_unfitted():
    model = CAGRForecaster()
    with pytest.raises(ValueError, match="positive"):
        model.fit([0, 10, 20])
    assert not model.is_fitted


# ----- Fibonacci helpers and model ----------------------------------------------
def test_fibonacci_numbers_and_ratios_match_hand_values():
    assert fibonacci_numbers(7) == [1, 1, 2, 3, 5, 8, 13]
    np.testing.assert_allclose(fibonacci_ratios(1, 4), [1, 2, 1.5, 5 / 3])


def test_fibonacci_ratios_converge_to_golden_ratio():
    golden = (1 + math.sqrt(5)) / 2
    assert fibonacci_ratios(1, 40)[-1] == pytest.approx(golden, rel=1e-12)


def test_fibonacci_forecast_and_fitted_path():
    model = FibonacciRatioForecaster(start_index=1).fit([10, 20, 30])
    np.testing.assert_allclose(model.fitted_values(), [10, 10, 20])  # 10, x1, x2
    np.testing.assert_allclose(model.predict(2), [30, 60])           # 30, x1, x2


def test_fibonacci_rejects_invalid_start_index():
    for bad in (0, -3, 1.5, True):
        with pytest.raises(ValueError):
            FibonacciRatioForecaster(start_index=bad)


# ----- moving average, SES and seasonal naive (MP5) ------------------------------
def test_moving_average_uses_last_window_values():
    model = MovingAverageForecaster(window=3).fit([10, 20, 30, 40])
    np.testing.assert_allclose(model.predict(2), [30, 30])          # mean(20, 30, 40)
    fitted = model.fitted_values()
    assert np.isnan(fitted[:3]).all()
    assert fitted[3] == pytest.approx(20)                            # mean(10, 20, 30)


def test_ses_hand_calculation():
    # level: 10 -> 0.5*20 + 0.5*10 = 15 -> 0.5*30 + 0.5*15 = 22.5
    model = SESForecaster(alpha=0.5).fit([10, 20, 30])
    assert model.level == pytest.approx(22.5)
    np.testing.assert_allclose(model.predict(3), [22.5, 22.5, 22.5])
    np.testing.assert_allclose(model.fitted_values(), [10, 10, 15])


def test_ses_with_alpha_one_is_the_naive_forecast():
    assert SESForecaster(alpha=1.0).fit([3, 9, 4, 7]).predict(1)[0] == pytest.approx(7)


@pytest.mark.parametrize("bad", [0, -0.1, 1.5, True])
def test_ses_rejects_invalid_alpha(bad):
    with pytest.raises(ValueError):
        SESForecaster(alpha=bad)


def test_seasonal_naive_repeats_last_season():
    y = [1, 2, 3, 10, 20, 30]
    model = SeasonalNaiveForecaster(season=3).fit(y)
    np.testing.assert_allclose(model.predict(5), [10, 20, 30, 10, 20])


def test_seasonal_naive_is_exact_on_a_periodic_series():
    week = np.array([50, 52, 51, 55, 70, 45, 35], dtype=float)
    y = np.tile(week, 4)
    _, preds = walk_forward(SeasonalNaiveForecaster(7), y, first_target=7)
    np.testing.assert_allclose(preds, y[7:])


# ----- walk-forward evaluation ----------------------------------------------------
def test_walk_forward_moving_average_hand_values():
    y = np.arange(1, 11, dtype=float)                 # days 1..10
    targets, preds = walk_forward(MovingAverageForecaster(3), y, first_target=3)
    assert targets.tolist() == list(range(3, 10))     # days 4..10
    np.testing.assert_allclose(preds, [2, 3, 4, 5, 6, 7, 8])


def test_walk_forward_never_uses_future_values():
    y = np.array([35, 40, 42, 50, 55, 60, 48, 52, 47, 45], dtype=float)
    changed = y.copy()
    changed[-1] = 999.0                                # only the last day differs
    _, a = walk_forward(SESForecaster(0.3), y, first_target=3)
    _, b = walk_forward(SESForecaster(0.3), changed, first_target=3)
    np.testing.assert_allclose(a, b)                   # forecast of day 10 uses days 1-9 only


def test_walk_forward_rejects_invalid_origin():
    with pytest.raises(ValueError):
        walk_forward(MovingAverageForecaster(3), [1, 2, 3], first_target=3)


# ----- the interface contract, checked for every model -------------------------
@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_predict_before_fit_raises(model_cls):
    with pytest.raises(RuntimeError, match="fit"):
        model_cls().predict(3)


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_invalid_horizon_is_rejected(model_cls):
    model = model_cls().fit([100, 110, 121])
    with pytest.raises(ValueError):
        model.predict(0)
    for bad in (1.5, True):
        with pytest.raises(TypeError):
            model.predict(bad)


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_invalid_series_is_rejected(model_cls):
    for bad in ([], [100, float("nan"), 120], [[100, 110], [120, 130]]):
        with pytest.raises(ValueError):
            model_cls().fit(bad)


def test_linear_needs_at_least_two_points():
    with pytest.raises(ValueError, match="at least 2"):
        LinearTrendForecaster().fit([100])


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_model_is_not_affected_by_later_changes_to_the_input(model_cls):
    data = np.array([100.0, 110.0, 121.0, 133.1])
    model = model_cls().fit(data)
    before = model.predict(2).copy()
    data[:] = 999.0                               # caller mutates the original array
    np.testing.assert_allclose(model.predict(2), before)


def test_clone_is_unfitted_and_keeps_settings():
    original = FibonacciRatioForecaster(start_index=5).fit([1, 2, 3])
    copy_ = original.clone()
    assert not copy_.is_fitted
    assert copy_.start_index == 5
    assert original.is_fitted


def test_repr_shows_settings_and_state():
    assert repr(FibonacciRatioForecaster(3)) == "FibonacciRatioForecaster(start_index=3, fitted=False)"
    assert repr(SESForecaster(0.25)) == "SESForecaster(alpha=0.25, fitted=False)"


# ----- bootstrap prediction intervals -------------------------------------------
def test_bootstrap_collapses_to_point_forecast_when_fit_is_perfect():
    y = 5 + 2 * np.arange(8)                      # zero residuals -> zero uncertainty
    point, lower, upper = bootstrap_intervals(
        LinearTrendForecaster(), y, 3, rng=np.random.default_rng(0), n_boot=200
    )
    np.testing.assert_allclose(lower, point, atol=1e-8)
    np.testing.assert_allclose(upper, point, atol=1e-8)


def test_bootstrap_is_reproducible_with_same_seed():
    noise = np.random.default_rng(1).normal(0, 5, size=10)
    y = 100 + 3 * np.arange(10) + noise
    run_a = bootstrap_intervals(LinearTrendForecaster(), y, 5, rng=np.random.default_rng(7))
    run_b = bootstrap_intervals(LinearTrendForecaster(), y, 5, rng=np.random.default_rng(7))
    for a, b in zip(run_a, run_b):
        np.testing.assert_array_equal(a, b)
    point, lower, upper = run_a
    assert np.all(lower <= upper)
    assert lower.shape == upper.shape == point.shape == (5,)


def test_bootstrap_requires_a_generator():
    with pytest.raises(TypeError):
        bootstrap_intervals(LinearTrendForecaster(), [1, 2, 3], 2)


def test_centring_has_no_effect_when_residuals_already_average_zero():
    # Least-squares residuals of a line with an intercept have mean zero.
    noise = np.random.default_rng(3).normal(0, 4, size=10)
    y = 50 + 2 * np.arange(10) + noise
    centred = bootstrap_intervals(LinearTrendForecaster(), y, 3, rng=np.random.default_rng(5))
    raw = bootstrap_intervals(
        LinearTrendForecaster(), y, 3, rng=np.random.default_rng(5), center=False
    )
    for a, b in zip(centred, raw):
        np.testing.assert_allclose(a, b)


def test_centred_cagr_interval_contains_point_forecast():
    # Curved series whose CAGR residuals have a clearly negative mean.
    y = [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800]
    assert CAGRForecaster().fit(y).residuals().mean() < -1
    point, lower, upper = bootstrap_intervals(
        CAGRForecaster(), y, 5, rng=np.random.default_rng(2026), n_boot=2000
    )
    assert np.all(lower < point) and np.all(point < upper)
