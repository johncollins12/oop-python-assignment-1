"""Tests for a1.microgrid: well-posedness, solving, feasibility, cost, data and input."""

import math

import numpy as np
import pandas as pd
import pytest

from a1.microgrid import (
    HybridMicroGrid,
    MicroGrid,
    generate_demand,
    load_demand_csv,
    parse_demand,
    read_demands_interactive,
    save_demand_csv,
)


@pytest.fixture
def grid() -> MicroGrid:
    return MicroGrid()


# ----- well-posedness -----------------------------------------------------------------
def test_determinant_and_condition_number_hand_values(grid):
    assert grid.determinant() == pytest.approx(3 * 1 - 2 * 4)        # -5
    assert grid.condition_number() == pytest.approx(3 + 2 * math.sqrt(2))
    assert grid.is_well_posed()


def test_singular_system_is_refused():
    singular = MicroGrid([[1, 2], [2, 4]])
    assert not singular.is_well_posed()
    with pytest.raises(np.linalg.LinAlgError):
        singular.solve_day(10, 20)


def test_non_square_coefficients_are_rejected():
    with pytest.raises(ValueError):
        MicroGrid([[1, 2, 3], [4, 5, 6]], sources=("a", "b"), costs=(1, 2))


# ----- solving --------------------------------------------------------------------------
def test_solve_day_hand_calculation(grid):
    # x = (2*D2 - D1)/5 = 28 ; y = (4*D1 - 3*D2)/5 = 8
    np.testing.assert_allclose(grid.solve_day(100, 120), [28, 8])


def test_loop_and_vectorised_solutions_agree(grid):
    D = generate_demand(30, rng=np.random.default_rng(1))[["D1_kWh", "D2_kWh"]].to_numpy().T
    np.testing.assert_allclose(grid.solve_loop(D), grid.solve_all(D))
    np.testing.assert_allclose(grid.A @ grid.solve_all(D), D)


@pytest.mark.parametrize("demands", [(-10, 120), (100, float("nan")), (100,)])
def test_invalid_demands_are_rejected(grid, demands):
    with pytest.raises(ValueError):
        grid.solve_day(*demands)


# ----- infeasible days ------------------------------------------------------------------
def test_infeasible_day_is_flagged(grid):
    D = np.array([[100, 100], [120, 140]], dtype=float)      # second day: y = -4
    S = grid.solve_all(D)
    np.testing.assert_allclose(S[:, 1], [36, -4])
    assert grid.infeasible(S).tolist() == [False, True]


def test_nnls_matches_exact_solution_on_feasible_days_and_stays_non_negative(grid):
    D = np.array([[100, 100], [120, 140]], dtype=float)
    S, residual = grid.solve_nonnegative(D)
    np.testing.assert_allclose(S[:, 0], [28, 8], atol=1e-9)
    assert residual[0] == pytest.approx(0, abs=1e-9)
    assert np.all(S >= 0) and residual[1] > 0


def test_clipping_sets_negatives_to_zero(grid):
    np.testing.assert_allclose(grid.clip_negative([[36, 5], [-4, 2]]), [[36, 5], [0, 2]])


# ----- cost ---------------------------------------------------------------------------------
def test_daily_cost_hand_calculation(grid):
    assert grid.daily_cost(np.array([28, 8])) == pytest.approx(28 * 150 + 8 * 450)   # 7,800


# ----- hybrid (extension) ------------------------------------------------------------------------
def test_hybrid_solves_three_sources():
    hybrid = HybridMicroGrid()
    assert hybrid.size == 3 and hybrid.determinant() == pytest.approx(-4)
    s = hybrid.solve_day(100, 120, 40)
    np.testing.assert_allclose(hybrid.A @ s, [100, 120, 40])


def test_hybrid_detects_linearly_dependent_constraint():
    # third row = first row + second row -> rank 2
    dependent = HybridMicroGrid(coefficients=[[3, 2, 1], [4, 1, 1], [7, 3, 2]])
    assert dependent.dependent_rows()
    assert dependent.condition_number() > 1e12
    with pytest.raises(np.linalg.LinAlgError):
        dependent.solve_day(100, 120, 220)


# ----- data -----------------------------------------------------------------------------------
def test_generated_demand_is_reproducible_with_weekly_pattern():
    a = generate_demand(28, rng=np.random.default_rng(5))
    b = generate_demand(28, rng=np.random.default_rng(5))
    pd.testing.assert_frame_equal(a, b)
    weekend = a["weekday"].isin(["Sat", "Sun"])
    assert a.loc[weekend, "D1_kWh"].mean() < a.loc[~weekend, "D1_kWh"].mean()


def test_csv_round_trip(tmp_path):
    df = generate_demand(30, rng=np.random.default_rng(2))
    path = save_demand_csv(df, tmp_path / "demand.csv")
    D = load_demand_csv(path)
    assert D.shape == (2, 30)
    np.testing.assert_allclose(D[0], df["D1_kWh"])


def test_csv_with_missing_column_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"D1_kWh": [100.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        load_demand_csv(path)


# ----- interactive input ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, reason", [("", "empty"), ("abc", "not a number"), ("-5", "negative")])
def test_parse_demand_rejects_bad_input(text, reason):
    with pytest.raises(ValueError, match=reason):
        parse_demand(text)


def test_interactive_reader_reprompts_until_valid():
    answers = iter(["", "abc", "-5", "100", "120"])
    messages = []
    values = read_demands_interactive(input_fn=lambda prompt: next(answers), print_fn=messages.append)
    assert values == [100.0, 120.0]
    assert len(messages) == 3                     # one message per rejected entry


def test_interactive_reader_gives_up_after_max_attempts():
    with pytest.raises(ValueError, match="attempts"):
        read_demands_interactive(input_fn=lambda prompt: "x", print_fn=lambda m: None, max_attempts=3)
