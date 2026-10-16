"""Mini-Project 3: Lake Victoria fish stock, prices and export-revenue risk (Jinja cooperative).

Stock follows discrete logistic growth with proportional harvesting:

    N(t+1) = N(t) + r N(t) (1 - N(t)/K) - h N(t)

Weekly revenue (UGX) = harvest (tonnes) x 1,000 kg/tonne x price (UGX/kg).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["FishStock", "PriceModel", "RiskAssessor", "weekly_revenue", "closed_season_mask", "KG_PER_TONNE"]

KG_PER_TONNE = 1000.0


class FishStock:
    """Logistic fish stock with a proportional weekly harvest rate.

    Parameters
    ----------
    r : intrinsic growth rate per week.
    K : carrying capacity (tonnes).
    n0 : initial stock (tonnes).
    h : fraction of the stock harvested each open week, 0 <= h < 1.
    """

    def __init__(self, r: float = 0.4, K: float = 10_000.0, n0: float = 4_000.0, h: float = 0.1) -> None:
        if r <= 0 or K <= 0:
            raise ValueError("r and K must be positive")
        if not 0 <= n0 <= K:
            raise ValueError("n0 must lie between 0 and K")
        if not 0 <= h < 1:
            raise ValueError("harvest rate h must satisfy 0 <= h < 1")
        self.r, self.K, self.n0, self.h = float(r), float(K), float(n0), float(h)

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.K:,.0f} t, N0={self.n0:,.0f} t, h={self.h})"

    def simulate(self, weeks: int = 52, open_weeks: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Simulate `weeks` weeks.

        `open_weeks` is an optional boolean array (length `weeks`); harvesting is
        suspended where it is False (closed season).

        Returns (stock of length weeks + 1, starting with N0; harvest per week of length weeks), in tonnes.
        """
        if weeks < 1:
            raise ValueError("weeks must be at least 1")
        open_ = np.ones(weeks, dtype=bool) if open_weeks is None else np.asarray(open_weeks, dtype=bool)
        if open_.shape != (weeks,):
            raise ValueError("open_weeks must have one value per week")
        stock = np.empty(weeks + 1)
        harvest = np.empty(weeks)
        stock[0] = self.n0
        for t in range(weeks):
            n = stock[t]
            harvest[t] = self.h * n if open_[t] else 0.0
            stock[t + 1] = max(0.0, n + self.r * n * (1.0 - n / self.K) - harvest[t])
        return stock, harvest

    # ----- theory --------------------------------------------------------------------
    def equilibrium_stock(self) -> float:
        """Long-run stock N* = K (1 - h/r) when h < r, otherwise 0 (collapse)."""
        return max(0.0, self.K * (1.0 - self.h / self.r))

    def sustainable_yield(self) -> float:
        """Weekly harvest at equilibrium, h N* (tonnes per week)."""
        return self.h * self.equilibrium_stock()

    def msy(self) -> float:
        """Maximum sustainable yield rK/4 (tonnes per week), reached at h = r/2 and N = K/2."""
        return self.r * self.K / 4.0


class PriceModel:
    """Weekly fish price (UGX/kg) as a seeded random walk kept within bounds.

    price_{t+1} = price_t + Normal(0, step_sd), reflected back inside [low, high].
    Reflection (rather than clipping) avoids piling probability exactly on the bounds.
    """

    def __init__(self, start: float = 12_000.0, low: float = 9_000.0, high: float = 16_000.0,
                 step_sd: float = 300.0) -> None:
        if not low < high:
            raise ValueError("low must be below high")
        if not low <= start <= high:
            raise ValueError("start must lie within the bounds")
        if step_sd < 0 or step_sd > (high - low):
            raise ValueError("step_sd must be between 0 and the width of the price band")
        self.start, self.low, self.high, self.step_sd = float(start), float(low), float(high), float(step_sd)

    def __repr__(self) -> str:
        return (f"PriceModel(start=UGX {self.start:,.0f}/kg, bounds={self.low:,.0f}-{self.high:,.0f}, "
                f"step_sd={self.step_sd:,.0f})")

    def _reflect(self, p: np.ndarray) -> np.ndarray:
        width = self.high - self.low
        q = np.mod(p - self.low, 2 * width)
        return self.low + np.where(q > width, 2 * width - q, q)

    def simulate_paths(self, n_paths: int, weeks: int, *, rng: np.random.Generator) -> np.ndarray:
        """Simulate `n_paths` price paths of length `weeks` (week 1 price = start). Shape (n_paths, weeks)."""
        if n_paths < 1 or weeks < 1:
            raise ValueError("n_paths and weeks must be at least 1")
        prices = np.empty((n_paths, weeks))
        prices[:, 0] = self.start
        steps = rng.normal(0.0, self.step_sd, size=(n_paths, weeks - 1))
        for t in range(1, weeks):
            prices[:, t] = self._reflect(prices[:, t - 1] + steps[:, t - 1])
        return prices

    def simulate(self, weeks: int, *, rng: np.random.Generator) -> np.ndarray:
        """One price path of length `weeks`."""
        return self.simulate_paths(1, weeks, rng=rng)[0]


def weekly_revenue(harvest_tonnes: ArrayLike, prices_ugx_per_kg: ArrayLike) -> np.ndarray:
    """Revenue (UGX) = harvest (t) x 1,000 kg/t x price (UGX/kg); broadcasts over price paths."""
    return np.asarray(harvest_tonnes, dtype=float) * KG_PER_TONNE * np.asarray(prices_ugx_per_kg, dtype=float)


@dataclass(frozen=True)
class RiskAssessor:
    """Classify revenue risk by the coefficient of variation (CV = sd / mean).

    Default thresholds, justified by the downside they imply: for roughly normal
    revenue, a 1-in-20 bad outcome lies about 1.645 sd below the mean, i.e. a
    shortfall of about 1.645 x CV. CV < 5 % keeps that shortfall under ~8 %
    ("low"); CV >= 15 % implies a shortfall of ~25 % or more ("high").
    """

    low_below: float = 0.05
    high_from: float = 0.15

    def __post_init__(self) -> None:
        if not 0 < self.low_below < self.high_from:
            raise ValueError("need 0 < low_below < high_from")

    @staticmethod
    def cv(values: ArrayLike) -> float:
        v = np.asarray(values, dtype=float)
        if v.size < 2:
            raise ValueError("at least two values are needed")
        mean = v.mean()
        if mean <= 0:
            raise ValueError("CV needs a positive mean")
        return float(v.std(ddof=1) / mean)

    def classify(self, cv: float) -> str:
        if cv < 0:
            raise ValueError("CV cannot be negative")
        if cv < self.low_below:
            return "low"
        if cv < self.high_from:
            return "moderate"
        return "high"

    @staticmethod
    def value_at_risk(values: ArrayLike, level: float = 0.05) -> dict[str, float]:
        """5 % VaR of an outcome distribution.

        Returns the 5th percentile ("quantile") and the shortfall below the mean
        ("shortfall" = mean - quantile), both in the units of `values`.
        """
        v = np.asarray(values, dtype=float)
        if v.size == 0 or not 0 < level < 1:
            raise ValueError("need non-empty values and 0 < level < 1")
        q = float(np.quantile(v, level))
        return {"quantile": q, "shortfall": float(v.mean() - q)}

    def assess(self, values: ArrayLike, level: float = 0.05) -> dict[str, float | str]:
        cv = self.cv(values)
        return {"mean": float(np.mean(values)), "cv": cv, "risk": self.classify(cv),
                **{f"VaR_{k}": val for k, val in self.value_at_risk(values, level).items()}}


def closed_season_mask(weeks: int, closed: Sequence[int], period: int = 52) -> np.ndarray:
    """Boolean mask, True = open week. `closed` lists week numbers (1-based) closed in every year."""
    if any(not 1 <= w <= period for w in closed):
        raise ValueError("closed weeks must be between 1 and the period")
    week_in_year = np.arange(weeks) % period + 1
    return ~np.isin(week_in_year, list(closed))

