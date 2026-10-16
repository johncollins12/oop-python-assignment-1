"""Mini-Project 5: taxi (matatu) route revenue, fare equilibrium and fleet planning."""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike

from a1.forecasting import Forecaster, SESForecaster, walk_forward
from a1.metrics import mae

__all__ = [
    "Route",
    "LinearMarket",
    "walk_forward_mae",
    "tune_ses_alpha",
    "vehicles_needed",
    "simulate_weekly_demand",
    "WEEKDAY_FACTORS",
]


class Route:
    """Daily passenger counts and the fare for one taxi route.

    Counts are stored as a read-only NumPy float array; day 1 is index 0.
    """

    def __init__(self, name: str, passengers: ArrayLike, fare: float) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        counts = np.asarray(passengers, dtype=float)
        if counts.ndim != 1 or counts.size == 0:
            raise ValueError("passengers must be a non-empty 1-D sequence")
        if not np.all(np.isfinite(counts)):
            raise ValueError("passengers must not contain NaN or infinite values")
        if np.any(counts < 0):
            raise ValueError("passenger counts must not be negative")
        if isinstance(fare, bool) or not isinstance(fare, (int, float)) or fare <= 0:
            raise ValueError("fare must be a positive number (UGX)")
        self._name = name.strip()
        self._passengers = counts.copy()
        self._passengers.setflags(write=False)
        self._fare = float(fare)

    @property
    def name(self) -> str:
        return self._name

    @property
    def passengers(self) -> np.ndarray:
        return self._passengers

    @property
    def fare(self) -> float:
        """Fare per passenger in UGX."""
        return self._fare

    @property
    def days(self) -> np.ndarray:
        """Day numbers 1..n."""
        return np.arange(1, len(self) + 1)

    def __len__(self) -> int:
        return int(self._passengers.size)

    def __repr__(self) -> str:
        return f"Route({self._name!r}, days={len(self)}, fare=UGX {self._fare:,.0f})"

    def daily_revenue(self) -> np.ndarray:
        """Revenue per day in UGX (passengers x fare)."""
        return self._passengers * self._fare

    def total_revenue(self) -> float:
        """Total revenue over all days in UGX."""
        return float(self.daily_revenue().sum())

    def stats(self) -> dict[str, float]:
        """Mean, median, sample variance and sample stdev of daily passengers (`statistics`)."""
        if len(self) < 2:
            raise ValueError("at least two days are needed for variance")
        values = self._passengers.tolist()
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "variance": statistics.variance(values),
            "stdev": statistics.stdev(values),
        }


@dataclass(frozen=True)
class LinearMarket:
    """Linear demand Qd = a - b*P and supply Qs = c + d*P (P in UGX, Q per trip-hour)."""

    demand_intercept: float
    demand_slope: float
    supply_intercept: float
    supply_slope: float

    def demand(self, price: float) -> float:
        return self.demand_intercept - self.demand_slope * price

    def supply(self, price: float) -> float:
        return self.supply_intercept + self.supply_slope * price

    def excess_demand(self, price: float) -> float:
        """Qd - Qs at `price`; positive means a shortage of seats."""
        return self.demand(price) - self.supply(price)

    def system(self) -> tuple[np.ndarray, np.ndarray]:
        """The equilibrium as A @ [P, Q] = b.

        Demand:  b*P + Q = a
        Supply: -d*P + Q = c
        """
        A = np.array([[self.demand_slope, 1.0], [-self.supply_slope, 1.0]])
        rhs = np.array([self.demand_intercept, self.supply_intercept])
        return A, rhs

    def equilibrium(self) -> tuple[float, float]:
        """Solve for (P*, Q*) with scipy.linalg.solve.

        Raises ValueError if the lines are parallel (determinant zero).
        """
        A, rhs = self.system()
        if math.isclose(np.linalg.det(A), 0.0, abs_tol=1e-12):
            raise ValueError("demand and supply are parallel: no unique equilibrium")
        price, quantity = scipy.linalg.solve(A, rhs)
        return float(price), float(quantity)


def walk_forward_mae(forecaster: Forecaster, values: ArrayLike, first_target_day: int = 4) -> float:
    """MAE of one-step-ahead walk-forward forecasts for days first_target_day..n (1-based)."""
    y = np.asarray(values, dtype=float)
    targets, preds = walk_forward(forecaster, y, first_target_day - 1)
    return mae(y[targets], preds)


def tune_ses_alpha(
    values: ArrayLike,
    alphas: Sequence[float] = tuple(np.round(np.arange(0.05, 1.0001, 0.05), 2)),
    first_target_day: int = 4,
) -> tuple[float, dict[float, float]]:
    """Grid-search alpha for SES by walk-forward MAE.

    Returns (best alpha, {alpha: MAE}). Ties go to the smallest alpha.
    Note: alpha is chosen on the same days it is scored on, so the tuned
    MAE is optimistic.
    """
    if len(alphas) == 0:
        raise ValueError("alphas must not be empty")
    scores = {float(a): walk_forward_mae(SESForecaster(float(a)), values, first_target_day)
              for a in alphas}
    best = min(scores, key=lambda a: (scores[a], a))
    return best, scores


def vehicles_needed(
    passengers: float,
    trips_per_day: int = 8,
    seats: int = 14,
    buffer: float = 0.15,
    minimum: int = 1,
) -> int:
    """Vehicles = ceil(passengers * (1 + buffer) / (trips_per_day * seats)), at least `minimum`.

    The value is rounded to 9 decimals before ceil so floating-point noise
    (e.g. 2.0000000000000004) does not add a vehicle.
    """
    if passengers < 0:
        raise ValueError("passengers must not be negative")
    if trips_per_day <= 0 or seats <= 0:
        raise ValueError("trips_per_day and seats must be positive")
    if buffer < 0:
        raise ValueError("buffer must not be negative")
    needed = passengers * (1.0 + buffer) / (trips_per_day * seats)
    return max(minimum, math.ceil(round(needed, 9)))


# Illustrative weekly pattern, Monday..Sunday: busier Friday, quieter Sunday.
WEEKDAY_FACTORS = (1.00, 0.95, 1.00, 1.05, 1.35, 1.10, 0.65)


def simulate_weekly_demand(
    n_days: int,
    base: float,
    *,
    rng: np.random.Generator,
    weekday_factors: Sequence[float] = WEEKDAY_FACTORS,
    noise_sd: float = 3.0,
) -> np.ndarray:
    """Simulate daily passenger counts with a weekly pattern (day 1 = Monday).

    count_t = round(base * factor[weekday(t)] + Normal(0, noise_sd)), floored at 0.
    """
    if n_days < 1:
        raise ValueError("n_days must be at least 1")
    if base <= 0 or noise_sd < 0:
        raise ValueError("base must be positive and noise_sd non-negative")
    if len(weekday_factors) != 7 or any(f < 0 for f in weekday_factors):
        raise ValueError("weekday_factors must be 7 non-negative numbers")
    factors = np.asarray(weekday_factors, dtype=float)[np.arange(n_days) % 7]
    counts = base * factors + rng.normal(0.0, noise_sd, size=n_days)
    return np.clip(np.round(counts), 0, None)
