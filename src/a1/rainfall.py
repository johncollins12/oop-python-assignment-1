"""Mini-Project 4: rainfall patterns, crop suitability and rainy-season detection.

Monthly rainfall is in mm, January to December.
"""

import calendar
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.signal
from numpy.typing import ArrayLike

__all__ = [
    "MONTHS",
    "Region",
    "CropRule",
    "cosine_similarity",
    "similarity_matrices",
    "detect_rainy_seasons",
    "parse_power_monthly",
]

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


class Region:
    """Twelve monthly rainfall totals (mm) for one region, stored as a read-only NumPy array."""

    def __init__(self, name: str, rainfall: ArrayLike) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        r = np.asarray(rainfall, dtype=float)
        if r.shape != (12,):
            raise ValueError("rainfall must contain exactly 12 monthly values (Jan-Dec)")
        if not np.all(np.isfinite(r)) or np.any(r < 0):
            raise ValueError("rainfall values must be finite and non-negative")
        self._name = name.strip()
        self._rain = r.copy()
        self._rain.setflags(write=False)

    @property
    def name(self) -> str:
        return self._name

    @property
    def rainfall(self) -> np.ndarray:
        return self._rain

    def __len__(self) -> int:
        return 12

    def __repr__(self) -> str:
        return f"Region({self._name!r}, annual={self.annual_total():.0f} mm)"

    def annual_total(self) -> float:
        return float(self._rain.sum())

    def mean(self) -> float:
        """Mean monthly rainfall (mm/month)."""
        return float(self._rain.mean())

    def wettest_month(self) -> tuple[str, float]:
        i = int(np.argmax(self._rain))
        return MONTHS[i], float(self._rain[i])

    def driest_month(self) -> tuple[str, float]:
        i = int(np.argmin(self._rain))
        return MONTHS[i], float(self._rain[i])

    def cv(self) -> float:
        """Coefficient of variation across months (sample sd / mean): how seasonal the rain is."""
        if self.mean() == 0:
            raise ValueError("CV is undefined for a region with no rain")
        return float(self._rain.std(ddof=1) / self.mean())


@dataclass(frozen=True)
class CropRule:
    """A crop's suitable monthly rainfall range (mm/month) with its source.

    Months below `min_mm` are a drought risk, above `max_mm` a waterlogging
    (excess-water) risk, and in between suitable.
    """

    crop: str
    min_mm: float
    max_mm: float
    source: str = ""
    derivation: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.min_mm < self.max_mm:
            raise ValueError("need 0 <= min_mm < max_mm")

    @classmethod
    def from_seasonal(cls, crop: str, season_min_mm: float, season_max_mm: float,
                      season_months: float, source: str = "") -> "CropRule":
        """Convert a seasonal water need (mm per growing period) to a monthly range."""
        if season_months <= 0:
            raise ValueError("season_months must be positive")
        return cls(crop, season_min_mm / season_months, season_max_mm / season_months, source,
                   f"{season_min_mm:g}-{season_max_mm:g} mm over {season_months:g} months")

    def classify(self, rainfall_mm: float) -> str:
        if rainfall_mm < 0 or not math.isfinite(rainfall_mm):
            raise ValueError("rainfall must be a finite, non-negative number")
        if rainfall_mm < self.min_mm:
            return "Drought risk"
        if rainfall_mm > self.max_mm:
            return "Waterlogging risk"
        return f"Good for {self.crop}"

    def score(self, rainfall_mm: float) -> int:
        """-1 drought risk, 0 suitable, +1 waterlogging risk (for heatmaps)."""
        label = self.classify(rainfall_mm)
        return {"Drought risk": -1, "Waterlogging risk": 1}.get(label, 0)


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """cos(theta) = a.b / (|a| |b|), computed with NumPy."""
    u, v = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if u.shape != v.shape or u.ndim != 1:
        raise ValueError("vectors must be 1-D and the same length")
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(u, v) / (nu * nv))


def similarity_matrices(regions: list[Region]) -> dict[str, pd.DataFrame]:
    """Pairwise cosine similarity, Pearson correlation and Euclidean distance (mm)."""
    names = [r.name for r in regions]
    X = np.array([r.rainfall for r in regions])
    n = len(regions)
    cos = np.array([[cosine_similarity(X[i], X[j]) for j in range(n)] for i in range(n)])
    pearson = np.corrcoef(X)
    euclid = np.array([[np.linalg.norm(X[i] - X[j]) for j in range(n)] for i in range(n)])
    frame = lambda m: pd.DataFrame(m, index=names, columns=names)
    return {"cosine similarity": frame(cos), "Pearson correlation": frame(pearson),
            "Euclidean distance (mm)": frame(euclid)}


def detect_rainy_seasons(rainfall: ArrayLike, min_prominence_frac: float = 0.15) -> dict:
    """Find rainy-season peaks with scipy.signal.find_peaks, treating the year as circular.

    The series is tiled three times so that December and January are neighbours;
    only peaks in the middle copy are kept. A peak must rise at least
    `min_prominence_frac` x (annual max - min) above its surroundings.

    Returns {"peaks": [month names], "prominences": [...], "regime": "unimodal"/"bimodal"/...}.
    """
    r = np.asarray(rainfall, dtype=float)
    if r.shape != (12,):
        raise ValueError("need 12 monthly values")
    if not 0 < min_prominence_frac < 1:
        raise ValueError("min_prominence_frac must be between 0 and 1")
    spread = r.max() - r.min()
    if spread == 0:
        return {"peaks": [], "prominences": [], "regime": "no seasonality"}
    tiled = np.tile(r, 3)
    idx, props = scipy.signal.find_peaks(tiled, prominence=min_prominence_frac * spread)
    keep = (idx >= 12) & (idx < 24)
    months = idx[keep] - 12
    prominences = props["prominences"][keep]
    regime = {0: "no clear season", 1: "unimodal", 2: "bimodal"}.get(len(months), "multimodal")
    return {"peaks": [MONTHS[m] for m in months], "prominences": prominences.round(1).tolist(),
            "regime": regime}


def parse_power_monthly(payload: dict, parameter: str = "PRECTOTCORR") -> pd.DataFrame:
    """Turn a NASA POWER monthly point response into monthly totals in mm.

    POWER reports monthly PRECTOTCORR as an average rate in mm/day, keyed
    'YYYYMM' (month 13 is the annual value and is skipped). Totals are
    rate x days in that calendar month. Fill values (-999) are rejected.
    """
    try:
        series = payload["properties"]["parameter"][parameter]
    except (KeyError, TypeError):
        raise ValueError(f"payload has no properties.parameter.{parameter}") from None
    rows = []
    for key, rate in series.items():
        year, month = int(key[:4]), int(key[4:])
        if month == 13:
            continue
        if rate is None or rate <= -999:
            raise ValueError(f"missing value for {key}")
        days = calendar.monthrange(year, month)[1]
        rows.append({"year": year, "month": month, "rain_mm": float(rate) * days})
    if not rows:
        raise ValueError("no monthly values found")
    return pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)
