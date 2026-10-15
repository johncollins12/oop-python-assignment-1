"""Tests for a1.population: validation, statistics, growth, study workflow and planning."""

import statistics

import numpy as np
import pytest

from a1.forecasting import CAGRForecaster, FibonacciRatioForecaster, LinearTrendForecaster
from a1.population import (
    DistrictForecastStudy,
    DistrictPopulation,
    classrooms_needed,
    geometric_interpolate,
)

YEARS = list(range(2015, 2025))
KAMPALA = [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800]


@pytest.fixture
def kampala() -> DistrictPopulation:
    return DistrictPopulation("Kampala", YEARS, KAMPALA)


# ----- construction and validation ----------------------------------------------
def test_len_and_repr(kampala):
    assert len(kampala) == 10
    assert repr(kampala) == "DistrictPopulation('Kampala', years=2015-2024, n=10, unit='thousands')"


def test_data_is_stored_as_read_only_float_arrays(kampala):
    assert kampala.populations.dtype == float
    with pytest.raises(ValueError):
        kampala.populations[0] = 0.0


@pytest.mark.parametrize(
    "years, pops",
    [
        ([], []),                                   # empty
        ([2015, 2016, 2017], [1, 2]),               # unequal lengths
        ([2015, 2016], [100, -5]),                  # negative population
        ([2015, 2016], [100, float("nan")]),        # NaN
        ([2016, 2015], [100, 110]),                 # decreasing years
        ([2015, 2017], [100, 110]),                 # gap in years
        ([2015.5, 2016.5], [100, 110]),             # non-whole years
    ],
)
def test_invalid_series_is_rejected(years, pops):
    with pytest.raises(ValueError):
        DistrictPopulation("X", years, pops)


def test_blank_name_is_rejected():
    with pytest.raises(ValueError):
        DistrictPopulation("  ", [2015, 2016], [1, 2])


# ----- statistics ----------------------------------------------------------------
def test_statistics_and_numpy_agree_when_ddof_is_one(kampala):
    s = kampala.stats_statistics()
    n1 = kampala.stats_numpy(ddof=1)
    for key in ("mean", "median", "variance", "stdev"):
        assert s[key] == pytest.approx(n1[key])


def test_numpy_default_is_population_variance(kampala):
    n = len(kampala)
    assert kampala.stats_numpy()["variance"] == pytest.approx(
        statistics.variance(KAMPALA) * (n - 1) / n
    )


# ----- growth ----------------------------------------------------------------------
def test_cagr_matches_hand_calculation(kampala):
    assert kampala.cagr() == pytest.approx(1.5 ** (1 / 9) - 1)     # about 4.61 %


def test_yoy_growth_values(kampala):
    growth = kampala.yoy_growth()
    assert growth.size == 9
    assert growth[0] == pytest.approx(50 / 1200)


# ----- split and study ---------------------------------------------------------------
def test_split_puts_2015_to_2021_in_training(kampala):
    train, test = kampala.split(2021)
    assert train.years.tolist() == list(range(2015, 2022))
    assert test.years.tolist() == [2022, 2023, 2024]


def test_split_rejects_empty_side(kampala):
    with pytest.raises(ValueError):
        kampala.split(2030)


def test_study_never_uses_test_years_for_fitting():
    base = DistrictPopulation("A", YEARS, KAMPALA)
    changed = DistrictPopulation("A", YEARS, KAMPALA[:7] + [5000, 6000, 7000])
    models = [LinearTrendForecaster(), CAGRForecaster()]
    preds_a = DistrictForecastStudy(base, models).test_predictions()
    preds_b = DistrictForecastStudy(changed, models).test_predictions()
    for name in preds_a:
        np.testing.assert_allclose(preds_a[name], preds_b[name])


def test_study_picks_the_generating_model():
    exact_growth = 100 * 1.04 ** np.arange(10)          # pure CAGR series
    district = DistrictPopulation("G", YEARS, exact_growth)
    study = DistrictForecastStudy(
        district, [LinearTrendForecaster(), CAGRForecaster(), FibonacciRatioForecaster()]
    )
    assert study.best_model("RMSE") == "CAGR"
    assert study.backtest()["CAGR"]["RMSE"] == pytest.approx(0, abs=1e-9)


def test_study_rejects_duplicate_model_names(kampala):
    with pytest.raises(ValueError):
        DistrictForecastStudy(kampala, [CAGRForecaster(), CAGRForecaster()])


def test_final_forecast_covers_2025_to_2029(kampala):
    years, values = DistrictForecastStudy(kampala, [LinearTrendForecaster()]).final_forecast(5)
    assert years.tolist() == [2025, 2026, 2027, 2028, 2029]
    assert values.shape == (5,)


# ----- planning and interpolation ---------------------------------------------------------
def test_classrooms_hand_example():
    # growth of 10 thousand -> 0.18 * 10,000 = 1,800 children -> 1800 / 53 = 33.96 -> 34
    assert classrooms_needed(100, 110) == 34


def test_classrooms_ignores_floating_point_noise():
    # 0.18 * (53 * 29 / 180) * 1000 / 53 is exactly 29 in theory; floats give 29.000000000000004
    assert classrooms_needed(0, 53 * 29 / 180) == 29


def test_classrooms_zero_when_population_falls():
    assert classrooms_needed(110, 100) == 0


def test_classrooms_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        classrooms_needed(100, 110, class_size=0)
    with pytest.raises(ValueError):
        classrooms_needed(100, 110, share=1.5)


def test_geometric_interpolation_hits_both_endpoints_with_constant_growth():
    values = geometric_interpolate(522.3, 560.089, 3)
    assert values[0] == 522.3
    assert values[-1] == 560.089
    ratios = values[1:] / values[:-1]
    np.testing.assert_allclose(ratios, ratios[0])