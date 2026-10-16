"""Forecasting models shared by the mini-projects.

Every model follows the same interface, defined by the abstract base class
`Forecaster`: call `fit(values)` on an equally spaced series, then
`predict(horizon)` for the next `horizon` steps.
"""

import copy
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "Forecaster",
    "LinearTrendForecaster",
    "CAGRForecaster",
    "FibonacciRatioForecaster",
    "MovingAverageForecaster",
    "SESForecaster",
    "SeasonalNaiveForecaster",
    "fibonacci_numbers",
    "fibonacci_ratios",
    "bootstrap_intervals",
    "walk_forward",
]


class Forecaster(ABC):
    """Abstract base for forecasting models on an equally spaced series.

    The base class owns validation, the fitted-state check and residuals
    (template method pattern); subclasses implement only their mathematics
    in `_fit`, `_predict` and `_fitted`.
    """

    name: str = "Forecaster"
    min_points: int = 2

    def __init__(self) -> None:
        self._values: np.ndarray | None = None

    # ----- public interface -------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """True once `fit` has completed successfully."""
        return self._values is not None

    def fit(self, values: ArrayLike) -> "Forecaster":
        """Fit the model to a 1-D series and return `self` (allows chaining)."""
        y = np.asarray(values, dtype=float)
        if y.ndim != 1:
            raise ValueError("values must be one-dimensional")
        if y.size < self.min_points:
            raise ValueError(
                f"{self.name} needs at least {self.min_points} values, got {y.size}"
            )
        if not np.all(np.isfinite(y)):
            raise ValueError("values must not contain NaN or infinite values")
        self._fit(y)                # may raise; model stays unfitted if it does
        self._values = y.copy()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next `horizon` steps after the end of the fitted series."""
        self._require_fitted()
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
            raise TypeError("horizon must be an integer")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        return self._predict(int(horizon))

    def fitted_values(self) -> np.ndarray:
        """In-sample values implied by the model, aligned with the fitted series."""
        self._require_fitted()
        return self._fitted()

    def residuals(self) -> np.ndarray:
        """Actual minus fitted values over the fitted series."""
        return self._values - self.fitted_values()

    def clone(self) -> "Forecaster":
        """Return an unfitted copy with the same settings (used for refitting)."""
        new = copy.deepcopy(self)
        new._values = None
        return new

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self._params().items())
        sep = ", " if params else ""
        return f"{type(self).__name__}({params}{sep}fitted={self.is_fitted})"

    # ----- hooks for subclasses ---------------------------------------------
    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name}: call fit() before using the model")

    def _params(self) -> dict:
        """Constructor settings shown in repr(); override if a model has any."""
        return {}

    @abstractmethod
    def _fit(self, y: np.ndarray) -> None: ...

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray: ...

    @abstractmethod
    def _fitted(self) -> np.ndarray: ...


class LinearTrendForecaster(Forecaster):
    """Straight line y = a + b*t fitted by least squares (np.polyfit, degree 1).

    t = 0, 1, ..., n-1, which centres the problem and avoids the poor
    conditioning of fitting on raw calendar years.
    """

    name = "Linear trend"

    def _fit(self, y: np.ndarray) -> None:
        t = np.arange(y.size, dtype=float)
        slope, intercept = np.polyfit(t, y, 1)
        self.slope, self.intercept = float(slope), float(intercept)
        self._n = y.size

    def _predict(self, horizon: int) -> np.ndarray:
        t = np.arange(self._n, self._n + horizon, dtype=float)
        return self.intercept + self.slope * t

    def _fitted(self) -> np.ndarray:
        return self.intercept + self.slope * np.arange(self._n, dtype=float)


class CAGRForecaster(Forecaster):
    """Constant growth at the compound annual growth rate of the fitted series.

    rate = (last / first) ** (1 / (n - 1)) - 1, and forecasts are anchored on
    the last observed value: y_{T+h} = y_T * (1 + rate) ** h.
    """

    name = "CAGR"

    def _fit(self, y: np.ndarray) -> None:
        if np.any(y <= 0):
            raise ValueError("CAGR requires strictly positive values")
        periods = y.size - 1
        self.rate = float((y[-1] / y[0]) ** (1.0 / periods) - 1.0)
        self._first, self._last, self._n = float(y[0]), float(y[-1]), y.size

    def _predict(self, horizon: int) -> np.ndarray:
        steps = np.arange(1, horizon + 1, dtype=float)
        return self._last * (1.0 + self.rate) ** steps

    def _fitted(self) -> np.ndarray:
        return self._first * (1.0 + self.rate) ** np.arange(self._n, dtype=float)


def fibonacci_numbers(count: int) -> list[int]:
    """Return F(1)..F(count) = 1, 1, 2, 3, 5, ... as exact Python integers."""
    if count < 0:
        raise ValueError("count must be non-negative")
    numbers, a, b = [], 1, 1
    for _ in range(count):
        numbers.append(a)
        a, b = b, a + b
    return numbers


def fibonacci_ratios(start_index: int, count: int) -> np.ndarray:
    """Return F(k+1)/F(k) for k = start_index, ..., start_index + count - 1.

    With start_index=1 the ratios are 1/1, 2/1, 3/2, 5/3, ... -> golden ratio.
    """
    if start_index < 1:
        raise ValueError("start_index must be at least 1")
    fib = fibonacci_numbers(start_index + count)      # fib[k-1] == F(k)
    return np.array(
        [fib[k] / fib[k - 1] for k in range(start_index, start_index + count)],
        dtype=float,
    )


class FibonacciRatioForecaster(Forecaster):
    """Previous cohort's model: scale an anchor value by successive Fibonacci ratios.

    Forecast: y_{T+h} = y_T * prod of the first h ratios, starting at `start_index`.
    In-sample path: the same rule applied from the first observed value.
    No parameter is estimated from the data.
    """

    name = "Fibonacci ratio"
    min_points = 1

    def __init__(self, start_index: int = 1) -> None:
        super().__init__()
        if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 1:
            raise ValueError("start_index must be an integer >= 1")
        self.start_index = start_index

    def _params(self) -> dict:
        return {"start_index": self.start_index}

    def _fit(self, y: np.ndarray) -> None:
        self._first, self._last, self._n = float(y[0]), float(y[-1]), y.size

    def _predict(self, horizon: int) -> np.ndarray:
        return self._last * np.cumprod(fibonacci_ratios(self.start_index, horizon))

    def _fitted(self) -> np.ndarray:
        growth = np.cumprod(fibonacci_ratios(self.start_index, self._n - 1))
        return self._first * np.concatenate(([1.0], growth))


class MovingAverageForecaster(Forecaster):
    """Forecast every future step as the mean of the last `window` observations.

    In-sample values are one-step-ahead moving averages; the first `window`
    positions have no forecast and are NaN.
    """

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be an integer >= 1")
        self.window = window
        self.min_points = window
        self.name = f"{window}-day moving average"

    def _params(self) -> dict:
        return {"window": self.window}

    def _fit(self, y: np.ndarray) -> None:
        self.level = float(np.mean(y[-self.window:]))
        self._y = y.copy()

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level)

    def _fitted(self) -> np.ndarray:
        out = np.full(self._y.size, np.nan)
        for t in range(self.window, self._y.size):
            out[t] = self._y[t - self.window:t].mean()
        return out


class SESForecaster(Forecaster):
    """Simple exponential smoothing with smoothing constant `alpha` in (0, 1].

    level_0 = y_0, then level_t = alpha * y_t + (1 - alpha) * level_{t-1}.
    Every future step is forecast as the final level. alpha = 1 gives the
    naive forecast (last value); small alpha averages over a long history.
    """

    name = "SES"
    min_points = 1

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha <= 1:
            raise ValueError("alpha must be a number in (0, 1]")
        self.alpha = float(alpha)

    def _params(self) -> dict:
        return {"alpha": self.alpha}

    def _fit(self, y: np.ndarray) -> None:
        fitted = np.empty(y.size)
        level = y[0]
        fitted[0] = level                       # no forecast exists for the first point
        for t in range(1, y.size):
            fitted[t] = level                   # one-step-ahead forecast of y_t
            level = self.alpha * y[t] + (1.0 - self.alpha) * level
        self.level = float(level)
        self._in_sample = fitted

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level)

    def _fitted(self) -> np.ndarray:
        return self._in_sample.copy()


class SeasonalNaiveForecaster(Forecaster):
    """Repeat the last full season: y_{T+h} = y_{T+h-season} (weekly when season=7).

    In-sample values are y_{t-season}; the first `season` positions are NaN.
    """

    def __init__(self, season: int = 7) -> None:
        super().__init__()
        if isinstance(season, bool) or not isinstance(season, int) or season < 1:
            raise ValueError("season must be an integer >= 1")
        self.season = season
        self.min_points = season
        self.name = f"Seasonal naive ({season})"

    def _params(self) -> dict:
        return {"season": self.season}

    def _fit(self, y: np.ndarray) -> None:
        self._last_season = y[-self.season:].copy()
        self._y = y.copy()

    def _predict(self, horizon: int) -> np.ndarray:
        return self._last_season[np.arange(horizon) % self.season]

    def _fitted(self) -> np.ndarray:
        out = np.full(self._y.size, np.nan)
        out[self.season:] = self._y[:-self.season]
        return out


def walk_forward(
    forecaster: Forecaster, values: ArrayLike, first_target: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling-origin one-step-ahead forecasts.

    For each target position t = first_target, ..., n-1 (0-based), a fresh
    clone of the model is fitted on values[:t] only and forecasts values[t].

    Returns (target positions, predictions).
    """
    y = np.asarray(values, dtype=float)
    if not 1 <= first_target < y.size:
        raise ValueError("first_target must leave at least one training and one target value")
    targets = np.arange(first_target, y.size)
    predictions = np.array(
        [forecaster.clone().fit(y[:t]).predict(1)[0] for t in targets]
    )
    return targets, predictions


def bootstrap_intervals(
    forecaster: Forecaster,
    values: ArrayLike,
    horizon: int,
    *,
    rng: np.random.Generator,
    n_boot: int = 1000,
    level: float = 0.95,
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual-bootstrap prediction intervals, refitting the model each time.

    For each of `n_boot` replicates: build a pseudo-series (fitted values +
    resampled residuals), refit a clone of the model, forecast, and add a
    resampled residual to each future step. Percentiles of the replicates give
    the interval. `rng` is required so results are always reproducible.

    If `center` is True (default) the residuals are shifted to mean zero before
    resampling. Without this, a model whose fitted path is biased (e.g. CAGR,
    which passes exactly through the first and last points) would push every
    replicate in one direction and shift the whole interval.

    Returns
    -------
    (point, lower, upper), each of length `horizon`.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    y = np.asarray(values, dtype=float)

    base = forecaster.clone().fit(y)
    point = base.predict(horizon)
    fitted = base.fitted_values()
    resid = y - fitted
    if center:
        resid = resid - resid.mean()

    sims = np.empty((n_boot, horizon))
    for b in range(n_boot):
        pseudo = fitted + rng.choice(resid, size=resid.size, replace=True)
        model = forecaster.clone().fit(pseudo)
        sims[b] = model.predict(horizon) + rng.choice(resid, size=horizon, replace=True)

    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(sims, [tail, 1.0 - tail], axis=0)
    return point, lower, upper
