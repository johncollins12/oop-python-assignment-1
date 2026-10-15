"""Mini-Project 1: district population analysis, model comparison and classroom planning."""

import math
import statistics
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from a1.forecasting import CAGRForecaster, Forecaster
from a1.metrics import evaluate

__all__ = [
    "DistrictPopulation",
    "DistrictForecastStudy",
    "classrooms_needed",
    "geometric_interpolate",
]

METRICS = ("MAE", "RMSE", "MAPE")


class DistrictPopulation:
    """Annual population series for one district.

    Years and populations are stored as read-only NumPy arrays. Years must be
    consecutive (one value per year), because the forecasters assume an
    equally spaced series.

    Parameters
    ----------
    name : district name.
    years : consecutive calendar years.
    populations : one value per year, in `unit`.
    unit : unit of the populations (default "thousands").
    sources : optional label per observation, e.g. "UBOS projection".
    """

    def __init__(
        self,
        name: str,
        years: ArrayLike,
        populations: ArrayLike,
        *,
        unit: str = "thousands",
        sources: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        yrs = np.asarray(years, dtype=float)
        pops = np.asarray(populations, dtype=float)
        if yrs.ndim != 1 or pops.ndim != 1:
            raise ValueError("years and populations must be one-dimensional")
        if yrs.size == 0:
            raise ValueError("years and populations must not be empty")
        if yrs.size != pops.size:
            raise ValueError(
                f"length mismatch: {yrs.size} years vs {pops.size} populations"
            )
        if not (np.all(np.isfinite(yrs)) and np.all(np.isfinite(pops))):
            raise ValueError("years and populations must not contain NaN or infinite values")
        if np.any(yrs != np.round(yrs)):
            raise ValueError("years must be whole numbers")
        if np.any(np.diff(yrs) != 1):
            raise ValueError("years must be consecutive and increasing (step of 1)")
        if np.any(pops < 0):
            raise ValueError("populations must not be negative")
        if sources is None:
            sources = ["unspecified"] * yrs.size
        if len(sources) != yrs.size:
            raise ValueError("sources must have one label per year")

        self._name = name.strip()
        self._years = yrs.astype(int)
        self._populations = pops.copy()
        self._years.setflags(write=False)
        self._populations.setflags(write=False)
        self._unit = unit
        self._sources = tuple(str(s) for s in sources)

    # ----- read-only attributes ---------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def years(self) -> np.ndarray:
        return self._years

    @property
    def populations(self) -> np.ndarray:
        return self._populations

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def sources(self) -> tuple[str, ...]:
        return self._sources

    # ----- dunder methods ---------------------------------------------------
    def __len__(self) -> int:
        return int(self._years.size)

    def __repr__(self) -> str:
        return (
            f"DistrictPopulation({self._name!r}, years={self._years[0]}-{self._years[-1]}, "
            f"n={len(self)}, unit={self._unit!r})"
        )

    # ----- statistics -------------------------------------------------------
    def stats_statistics(self) -> dict[str, float]:
        """Mean, median, sample variance and sample stdev via the `statistics` module.

        `statistics.variance` divides by n - 1 (sample variance).
        """
        if len(self) < 2:
            raise ValueError("at least two values are needed for variance")
        values = self._populations.tolist()          # plain floats for `statistics`
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "variance": statistics.variance(values),
            "stdev": statistics.stdev(values),
        }

    def stats_numpy(self, ddof: int = 0) -> dict[str, float]:
        """Mean, median, variance and stdev via NumPy.

        NumPy divides by n - ddof: ddof=0 (default) gives the population
        variance, ddof=1 gives the sample variance and matches `statistics`.
        """
        if ddof not in (0, 1):
            raise ValueError("ddof must be 0 or 1")
        if len(self) - ddof < 1:
            raise ValueError("not enough values for this ddof")
        p = self._populations
        return {
            "mean": float(np.mean(p)),
            "median": float(np.median(p)),
            "variance": float(np.var(p, ddof=ddof)),
            "stdev": float(np.std(p, ddof=ddof)),
        }

    def yoy_growth(self) -> np.ndarray:
        """Year-on-year growth rates (P_t - P_{t-1}) / P_{t-1}, one per interval."""
        if len(self) < 2:
            raise ValueError("at least two values are needed for growth rates")
        previous = self._populations[:-1]
        if np.any(previous == 0):
            raise ValueError("growth rate is undefined after a zero population")
        return self._populations[1:] / previous - 1.0

    def cagr(self) -> float:
        """Compound annual growth rate over the whole series (len - 1 intervals).

        Reuses CAGRForecaster so the formula exists in one place only.
        """
        return CAGRForecaster().fit(self._populations).rate

    def split(self, last_train_year: int) -> tuple["DistrictPopulation", "DistrictPopulation"]:
        """Split into training (<= last_train_year) and test (> last_train_year) parts."""
        train_mask = self._years <= last_train_year
        if train_mask.all() or not train_mask.any():
            raise ValueError(
                f"last_train_year={last_train_year} leaves an empty training or test set"
            )
        return self._subset(train_mask), self._subset(~train_mask)

    def _subset(self, mask: np.ndarray) -> "DistrictPopulation":
        return DistrictPopulation(
            self._name,
            self._years[mask],
            self._populations[mask],
            unit=self._unit,
            sources=[s for s, keep in zip(self._sources, mask) if keep],
        )


class DistrictForecastStudy:
    """Compare forecasters on one district, choose the best, and forecast ahead.

    Composition: a study *has* a DistrictPopulation and several Forecasters.
    The forecasters passed in are never modified; fresh clones are fitted.
    """

    def __init__(
        self,
        district: DistrictPopulation,
        forecasters: Sequence[Forecaster],
        last_train_year: int = 2021,
    ) -> None:
        if not isinstance(district, DistrictPopulation):
            raise TypeError("district must be a DistrictPopulation")
        if not forecasters:
            raise ValueError("at least one forecaster is required")
        if not all(isinstance(f, Forecaster) for f in forecasters):
            raise TypeError("every forecaster must be a Forecaster instance")
        names = [f.name for f in forecasters]
        if len(set(names)) != len(names):
            raise ValueError("forecaster names must be unique")

        self.district = district
        self.forecasters = list(forecasters)
        self.last_train_year = last_train_year
        self.train, self.test = district.split(last_train_year)
        self._backtest: dict[str, dict] | None = None

    def __repr__(self) -> str:
        names = ", ".join(f.name for f in self.forecasters)
        return (
            f"DistrictForecastStudy({self.district.name!r}, train<={self.last_train_year}, "
            f"models=[{names}])"
        )

    def backtest(self) -> dict[str, dict[str, float]]:
        """Fit each model on the training years and score it on the test years.

        Returns {model name: {"MAE":..., "RMSE":..., "MAPE":...}}.
        """
        if self._backtest is None:
            results = {}
            horizon = len(self.test)
            for f in self.forecasters:
                model = f.clone().fit(self.train.populations)
                predicted = model.predict(horizon)
                results[f.name] = {
                    "model": model,
                    "predicted": predicted,
                    "metrics": evaluate(self.test.populations, predicted),
                }
            self._backtest = results
        return {name: r["metrics"] for name, r in self._backtest.items()}

    def test_predictions(self) -> dict[str, np.ndarray]:
        """Test-period predictions of each model (fitted on training years only)."""
        self.backtest()
        return {name: r["predicted"].copy() for name, r in self._backtest.items()}

    def train_fitted(self) -> dict[str, np.ndarray]:
        """In-sample fitted values of each model over the training years."""
        self.backtest()
        return {name: r["model"].fitted_values() for name, r in self._backtest.items()}

    def best_model(self, metric: str = "RMSE") -> str:
        """Name of the model with the lowest test error on `metric`.

        Ties are broken by the order in which the forecasters were supplied.
        """
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        scores = self.backtest()
        return min(scores, key=lambda name: scores[name][metric])

    def final_model(self, metric: str = "RMSE") -> Forecaster:
        """The best model refitted on all years (training + test)."""
        chosen = self.best_model(metric)
        template = next(f for f in self.forecasters if f.name == chosen)
        return template.clone().fit(self.district.populations)

    def final_forecast(self, horizon: int = 5, metric: str = "RMSE") -> tuple[np.ndarray, np.ndarray]:
        """Forecast `horizon` years after the last observed year with the final model.

        Returns (years, values).
        """
        values = self.final_model(metric).predict(horizon)
        last = int(self.district.years[-1])
        return np.arange(last + 1, last + 1 + horizon), values

    def classrooms_needed(self, target_year: int = 2029, metric: str = "RMSE", **kwargs) -> int:
        """Additional classrooms needed by `target_year`, relative to the last observed year.

        Extra keyword arguments are passed to `classrooms_needed()`.
        """
        last = int(self.district.years[-1])
        if target_year <= last:
            raise ValueError("target_year must be after the last observed year")
        years, values = self.final_forecast(target_year - last, metric)
        return classrooms_needed(self.district.populations[-1], values[-1], **kwargs)


def classrooms_needed(
    pop_base: float,
    pop_target: float,
    share: float = 0.18,
    class_size: int = 53,
    unit_multiplier: float = 1000.0,
) -> int:
    """Extra classrooms for the growth in primary-age children between two years.

    classrooms = ceil(share * (pop_target - pop_base) * unit_multiplier / class_size),
    floored at zero (a falling population needs no new classrooms).
    The value is rounded to 9 decimals before ceil so floating-point noise such as
    29.000000000000004 does not become 30.
    """
    if not 0 < share <= 1:
        raise ValueError("share must be in (0, 1]")
    if class_size <= 0:
        raise ValueError("class_size must be positive")
    if pop_base < 0 or pop_target < 0:
        raise ValueError("populations must not be negative")
    extra = share * (pop_target - pop_base) * unit_multiplier / class_size
    if extra <= 0:
        return 0
    return math.ceil(round(extra, 9))


def geometric_interpolate(start: float, end: float, periods: int) -> np.ndarray:
    """Values at constant growth from `start` to `end` over `periods` intervals.

    Returns periods + 1 values, the first equal to `start` and the last to `end`.
    """
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 1:
        raise ValueError("periods must be an integer >= 1")
    if start <= 0 or end <= 0:
        raise ValueError("start and end must be positive")
    ratio = (end / start) ** (1.0 / periods)
    values = start * ratio ** np.arange(periods + 1, dtype=float)
    values[-1] = end                                 # remove rounding drift at the endpoint
    return values