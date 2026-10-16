"""Tests for a1.fish_stock: logistic dynamics, prices, revenue units, risk rules and closed seasons."""

import numpy as np
import pytest

from a1.fish_stock import (
    FishStock,
    PriceModel,
    RiskAssessor,
    closed_season_mask,
    weekly_revenue,
)


# ----- stock dynamics ---------------------------------------------------------------------
def test_first_week_hand_calculation():
    # N1 = 4000 + 0.4*4000*(1 - 0.4) - 0.1*4000 = 4000 + 960 - 400 = 4560
    stock, harvest = FishStock(h=0.1).simulate(1)
    assert stock.tolist() == [4000, 4560]
    assert harvest[0] == pytest.approx(400)


def test_stock_converges_to_theoretical_equilibrium():
    for h in (0.05, 0.1, 0.2, 0.3):
        model = FishStock(h=h)
        stock, harvest = model.simulate(300)
        assert stock[-1] == pytest.approx(model.equilibrium_stock(), rel=1e-6)
        assert harvest[-1] == pytest.approx(model.sustainable_yield(), rel=1e-6)


def test_msy_and_where_it_is_reached():
    assert FishStock().msy() == pytest.approx(1000)               # rK/4 = 0.4 * 10,000 / 4
    yields = {h: FishStock(h=h).sustainable_yield() for h in (0.05, 0.1, 0.2, 0.3)}
    assert max(yields, key=yields.get) == 0.2                     # h = r/2
    assert yields[0.2] == pytest.approx(1000)


def test_without_harvest_stock_approaches_carrying_capacity():
    stock, harvest = FishStock(h=0.0).simulate(200)
    assert stock[-1] == pytest.approx(10_000, rel=1e-6)
    assert harvest.sum() == 0


def test_stock_never_becomes_negative():
    stock, _ = FishStock(h=0.99).simulate(52)
    assert np.all(stock >= 0)


@pytest.mark.parametrize("kwargs", [{"r": 0}, {"K": -1}, {"n0": -5}, {"n0": 20_000}, {"h": 1.0}, {"h": -0.1}])
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        FishStock(**kwargs)


def test_closed_season_stops_harvest():
    mask = closed_season_mask(104, closed=range(14, 22))
    assert mask.sum() == 104 - 16                                  # 8 weeks closed in each of 2 years
    _, harvest = FishStock(h=0.2).simulate(104, open_weeks=mask)
    assert np.all(harvest[~mask] == 0) and np.all(harvest[mask] > 0)


def test_closed_season_rejects_invalid_week():
    with pytest.raises(ValueError):
        closed_season_mask(52, closed=[0, 53])


# ----- prices and revenue -------------------------------------------------------------------
def test_price_paths_start_at_12000_stay_in_bounds_and_are_reproducible():
    model = PriceModel()
    a = model.simulate_paths(500, 52, rng=np.random.default_rng(1))
    b = model.simulate_paths(500, 52, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert np.all(a[:, 0] == 12_000)
    assert a.min() >= 9_000 and a.max() <= 16_000


def test_reflection_at_the_bounds():
    model = PriceModel()
    np.testing.assert_allclose(model._reflect(np.array([16_500.0, 8_500.0, 12_000.0])),
                               [15_500, 9_500, 12_000])


def test_zero_volatility_gives_a_constant_price():
    prices = PriceModel(step_sd=0).simulate(10, rng=np.random.default_rng(0))
    assert np.all(prices == 12_000)


def test_revenue_units():
    # 1 tonne = 1,000 kg at UGX 12,000/kg -> UGX 12,000,000
    assert weekly_revenue([1.0], [12_000])[0] == pytest.approx(12_000_000)


# ----- risk -----------------------------------------------------------------------------------
def test_cv_and_classification():
    ra = RiskAssessor()
    assert ra.cv([90, 100, 110]) == pytest.approx(0.1)            # sd 10 / mean 100
    assert [ra.classify(c) for c in (0.03, 0.10, 0.20)] == ["low", "moderate", "high"]


def test_cv_is_unit_free_but_variance_is_not():
    revenue_ugx = np.array([9.0e9, 1.0e10, 1.1e10])
    revenue_bn = revenue_ugx / 1e9
    assert RiskAssessor.cv(revenue_ugx) == pytest.approx(RiskAssessor.cv(revenue_bn))
    assert np.var(revenue_ugx, ddof=1) == pytest.approx(np.var(revenue_bn, ddof=1) * 1e18)


def test_value_at_risk_hand_values():
    values = np.arange(1, 101, dtype=float)                       # 5th percentile (linear) = 5.95
    var = RiskAssessor.value_at_risk(values, 0.05)
    assert var["quantile"] == pytest.approx(5.95)
    assert var["shortfall"] == pytest.approx(50.5 - 5.95)


def test_risk_assessor_rejects_invalid_input():
    with pytest.raises(ValueError):
        RiskAssessor(low_below=0.2, high_from=0.1)
    with pytest.raises(ValueError):
        RiskAssessor.cv([0, 0, 0])
