"""Tests for a1.rainfall: Region, CropRule, similarity measures, season detection and data parsing."""

import numpy as np
import pytest
import scipy.spatial.distance

from a1.rainfall import (
    CropRule,
    Region,
    cosine_similarity,
    detect_rainy_seasons,
    parse_power_monthly,
    similarity_matrices,
)

KAMPALA = [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]
GULU = [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15]
MBARARA = [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90]


# ----- Region ---------------------------------------------------------------------------
def test_region_summary_hand_values():
    r = Region("Kampala", KAMPALA)
    assert r.annual_total() == 1600
    assert r.mean() == pytest.approx(1600 / 12)
    assert r.wettest_month() == ("May", 220)
    assert r.driest_month() == ("Sep", 60)
    assert r.cv() == pytest.approx(np.std(KAMPALA, ddof=1) / np.mean(KAMPALA))
    assert len(r) == 12


@pytest.mark.parametrize("rain", [[100] * 11, [100] * 11 + [-5], [100] * 11 + [float("nan")]])
def test_invalid_region_is_rejected(rain):
    with pytest.raises(ValueError):
        Region("X", rain)


# ----- CropRule ---------------------------------------------------------------------------
def test_crop_rule_from_seasonal_need():
    maize = CropRule.from_seasonal("maize", 500, 800, 5)
    assert (maize.min_mm, maize.max_mm) == (100, 160)


def test_crop_rule_classification_and_boundaries():
    maize = CropRule("maize", 100, 160)
    assert maize.classify(60) == "Drought risk"
    assert maize.classify(100) == "Good for maize"
    assert maize.classify(160) == "Good for maize"
    assert maize.classify(220) == "Waterlogging risk"
    assert [maize.score(x) for x in (60, 120, 220)] == [-1, 0, 1]


def test_crop_rule_rejects_invalid_ranges():
    with pytest.raises(ValueError):
        CropRule("x", 200, 100)
    with pytest.raises(ValueError):
        CropRule("x", 100, 200).classify(-1)


# ----- similarity ---------------------------------------------------------------------------
def test_cosine_similarity_matches_scipy():
    expected = 1 - scipy.spatial.distance.cosine(KAMPALA, GULU)
    assert cosine_similarity(KAMPALA, GULU) == pytest.approx(expected)


def test_cosine_ignores_magnitude_but_euclidean_does_not():
    wetter = np.array(KAMPALA) * 2
    assert cosine_similarity(KAMPALA, wetter) == pytest.approx(1.0)
    m = similarity_matrices([Region("K", KAMPALA), Region("K2", wetter)])
    assert m["Euclidean distance (mm)"].loc["K", "K2"] == pytest.approx(np.linalg.norm(KAMPALA))


def test_similarity_matrices_are_symmetric_with_unit_diagonal():
    m = similarity_matrices([Region("K", KAMPALA), Region("G", GULU), Region("M", MBARARA)])
    for name in ("cosine similarity", "Pearson correlation"):
        np.testing.assert_allclose(m[name].values, m[name].values.T)
        np.testing.assert_allclose(np.diag(m[name].values), 1)


def test_cosine_rejects_zero_vector():
    with pytest.raises(ValueError):
        cosine_similarity([0] * 12, KAMPALA)


# ----- rainy-season detection -------------------------------------------------------------------
def test_clear_bimodal_and_unimodal_patterns():
    bimodal = [20, 40, 150, 200, 150, 40, 20, 30, 120, 180, 120, 30]
    unimodal = [5, 10, 40, 90, 150, 200, 220, 200, 150, 90, 30, 10]
    assert detect_rainy_seasons(bimodal)["regime"] == "bimodal"
    assert detect_rainy_seasons(unimodal)["regime"] == "unimodal"


def test_peak_across_the_year_end_is_found():
    # wettest month is December; without circular padding it would be missed
    rain = [150, 100, 40, 20, 10, 10, 10, 10, 20, 50, 120, 200]
    assert detect_rainy_seasons(rain)["peaks"] == ["Dec"]


def test_flat_rainfall_has_no_season():
    assert detect_rainy_seasons([80] * 12)["regime"] == "no seasonality"


# ----- NASA POWER parsing ---------------------------------------------------------------------------
def test_parse_power_monthly_converts_mm_per_day_to_monthly_totals():
    payload = {"properties": {"parameter": {"PRECTOTCORR": {
        "202001": 2.0, "202002": 1.0, "202013": 5.0}}}}       # 2020 is a leap year; month 13 = annual
    df = parse_power_monthly(payload)
    assert df["rain_mm"].tolist() == [62.0, 29.0]
    assert df["month"].tolist() == [1, 2]


def test_parse_power_monthly_rejects_fill_values_and_bad_payloads():
    with pytest.raises(ValueError, match="missing"):
        parse_power_monthly({"properties": {"parameter": {"PRECTOTCORR": {"202001": -999.0}}}})
    with pytest.raises(ValueError):
        parse_power_monthly({"unexpected": {}})
