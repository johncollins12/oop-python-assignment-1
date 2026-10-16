"""Tests for a1.taxi_routes: route revenue, equilibrium, tuning, fleet size and simulation."""

import statistics

import numpy as np
import pytest

from a1.forecasting import MovingAverageForecaster, SESForecaster
from a1.taxi_routes import (
    LinearMarket,
    Route,
    simulate_weekly_demand,
    tune_ses_alpha,
    vehicles_needed,
    walk_forward_mae,
)

NTINDA = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]


@pytest.fixture
def ntinda() -> Route:
    return Route("Kampala-Ntinda", NTINDA, 2000)


# ----- Route ----------------------------------------------------------------------
def test_revenue_hand_calculation(ntinda):
    assert ntinda.daily_revenue()[0] == 70_000          # 35 x 2,000
    assert ntinda.total_revenue() == 948_000            # 474 passengers x 2,000


def test_stats_match_statistics_module(ntinda):
    s = ntinda.stats()
    assert s["mean"] == pytest.approx(47.4)
    assert s["variance"] == pytest.approx(statistics.variance(NTINDA))


def test_len_repr_and_days(ntinda):
    assert len(ntinda) == 10
    assert repr(ntinda) == "Route('Kampala-Ntinda', days=10, fare=UGX 2,000)"
    assert ntinda.days.tolist() == list(range(1, 11))


@pytest.mark.parametrize(
    "passengers, fare",
    [([], 2000), ([10, -1], 2000), ([10, float("nan")], 2000), ([10, 20], 0), ([10, 20], -500)],
)
def test_invalid_route_is_rejected(passengers, fare):
    with pytest.raises(ValueError):
        Route("X", passengers, fare)


# ----- equilibrium ------------------------------------------------------------------
@pytest.fixture
def market() -> LinearMarket:
    return LinearMarket(120, 0.02, 10, 0.03)


def test_equilibrium_hand_calculation(market):
    # 120 - 0.02P = 10 + 0.03P  ->  110 = 0.05P  ->  P* = 2,200 ; Q* = 76
    price, quantity = market.equilibrium()
    assert price == pytest.approx(2200)
    assert quantity == pytest.approx(76)
    assert market.demand(price) == pytest.approx(market.supply(price))


def test_excess_demand_at_current_fare(market):
    assert market.demand(2000) == pytest.approx(80)
    assert market.supply(2000) == pytest.approx(70)
    assert market.excess_demand(2000) == pytest.approx(10)


def test_parallel_lines_have_no_equilibrium():
    with pytest.raises(ValueError, match="parallel"):
        LinearMarket(120, 0.02, 10, -0.02).equilibrium()


# ----- walk-forward MAE and alpha tuning ------------------------------------------------
def test_walk_forward_mae_hand_value():
    y = np.arange(1, 11, dtype=float)       # MA(3) forecast of day t is t - 2, error 2
    assert walk_forward_mae(MovingAverageForecaster(3), y, first_target_day=4) == pytest.approx(2)


def test_tuned_alpha_has_the_lowest_score():
    best, scores = tune_ses_alpha(NTINDA)
    assert best in scores
    assert scores[best] == min(scores.values())
    assert len(scores) == 20 and min(scores) == 0.05 and max(scores) == 1.0
    assert scores[best] == pytest.approx(walk_forward_mae(SESForecaster(best), NTINDA))


def test_tuning_rejects_empty_grid():
    with pytest.raises(ValueError):
        tune_ses_alpha(NTINDA, alphas=[])


# ----- fleet planning -----------------------------------------------------------------
def test_vehicles_hand_examples():
    assert vehicles_needed(50) == 1          # 57.5 / 112 = 0.51 -> 1
    assert vehicles_needed(200) == 3         # 230 / 112 = 2.05 -> 3
    assert vehicles_needed(0) == 1           # minimum of one vehicle


def test_vehicles_ignores_floating_point_noise():
    # 112 * 29 / 1.15 passengers need exactly 29 vehicles; floats give 29.000000000000004
    assert vehicles_needed(112 * 29 / 1.15) == 29


def test_vehicles_rejects_invalid_input():
    with pytest.raises(ValueError):
        vehicles_needed(-5)
    with pytest.raises(ValueError):
        vehicles_needed(50, seats=0)


# ----- weekly simulation ------------------------------------------------------------------
def test_simulation_is_reproducible_and_weekly():
    a = simulate_weekly_demand(60, 50, rng=np.random.default_rng(11))
    b = simulate_weekly_demand(60, 50, rng=np.random.default_rng(11))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (60,) and np.all(a >= 0)
    fridays, sundays = a[4::7], a[6::7]       # day 1 = Monday
    assert fridays.mean() > a.mean() > sundays.mean()


def test_simulation_without_noise_follows_factors_exactly():
    counts = simulate_weekly_demand(7, 100, rng=np.random.default_rng(0), noise_sd=0)
    np.testing.assert_array_equal(counts, [100, 95, 100, 105, 135, 110, 65])
