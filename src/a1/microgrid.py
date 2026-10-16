"""Mini-Project 2: solar micro-grid dispatch planning for a rural health centre (Kasese).

Each day the energy drawn from each source must satisfy a set of linear demand
constraints A @ s = d, e.g. for solar x and battery y:

    3x + 2y = D1   (daytime load, kWh)
    4x +  y = D2   (critical-equipment load, kWh)
"""

import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize
from numpy.typing import ArrayLike

__all__ = [
    "MicroGrid",
    "HybridMicroGrid",
    "generate_demand",
    "save_demand_csv",
    "load_demand_csv",
    "read_demands_interactive",
    "parse_demand",
]


class MicroGrid:
    """Square linear dispatch model: coefficient matrix A, one unknown per energy source.

    Parameters
    ----------
    coefficients : square matrix A (rows = constraints, columns = sources).
    sources : names of the sources, one per column.
    costs : cost of each source in UGX per kWh.
    """

    def __init__(
        self,
        coefficients: ArrayLike = ((3.0, 2.0), (4.0, 1.0)),
        sources: Sequence[str] = ("solar", "battery"),
        costs: Sequence[float] = (150.0, 450.0),
    ) -> None:
        A = np.asarray(coefficients, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
            raise ValueError("coefficients must be a non-empty square matrix")
        if not np.all(np.isfinite(A)):
            raise ValueError("coefficients must be finite")
        if len(sources) != A.shape[0] or len(costs) != A.shape[0]:
            raise ValueError("need one source name and one cost per column")
        if any(c < 0 for c in costs):
            raise ValueError("costs must not be negative")
        self._A = A.copy()
        self._A.setflags(write=False)
        self.sources = tuple(sources)
        self.costs = np.asarray(costs, dtype=float)

    # ----- the system ---------------------------------------------------------
    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def size(self) -> int:
        return self._A.shape[0]

    def determinant(self) -> float:
        return float(np.linalg.det(self._A))

    def condition_number(self) -> float:
        """2-norm condition number (np.linalg.cond); inf for a singular matrix."""
        return float(np.linalg.cond(self._A))

    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self._A))

    def is_well_posed(self, max_condition: float = 1e8) -> bool:
        """True if A has full rank and its condition number is below `max_condition`."""
        return self.rank() == self.size and self.condition_number() < max_condition

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(sources={self.sources}, det={self.determinant():.3g}, "
                f"cond={self.condition_number():.3g})")

    # ----- solving --------------------------------------------------------------
    def _check_demands(self, demands: ArrayLike) -> np.ndarray:
        D = np.asarray(demands, dtype=float)
        if D.shape[0] != self.size or D.ndim not in (1, 2):
            raise ValueError(f"demands must have {self.size} rows (one per constraint)")
        if not np.all(np.isfinite(D)):
            raise ValueError("demands must be finite numbers")
        if np.any(D < 0):
            raise ValueError("demands must not be negative")
        return D

    def _require_well_posed(self) -> None:
        if not self.is_well_posed():
            raise np.linalg.LinAlgError(
                f"system is singular or ill-conditioned (rank {self.rank()} of {self.size})"
            )

    def solve_day(self, *demands: float) -> np.ndarray:
        """Energy from each source for one day, e.g. solve_day(d1, d2) -> [x, y]."""
        d = self._check_demands(demands)
        self._require_well_posed()
        return scipy.linalg.solve(self._A, d)

    def solve_loop(self, demands: ArrayLike) -> np.ndarray:
        """Solve each day (column) separately in a Python loop; returns size x n_days."""
        D = self._check_demands(demands)
        self._require_well_posed()
        return np.column_stack([scipy.linalg.solve(self._A, D[:, j]) for j in range(D.shape[1])])

    def solve_all(self, demands: ArrayLike) -> np.ndarray:
        """Solve every day in one vectorised call (right-hand side size x n_days)."""
        D = self._check_demands(demands)
        self._require_well_posed()
        return scipy.linalg.solve(self._A, D)

    @staticmethod
    def infeasible(solution: ArrayLike, tol: float = 1e-9) -> np.ndarray:
        """Boolean per day: True where any source would have to supply negative energy."""
        S = np.asarray(solution, dtype=float)
        return np.any(S < -tol, axis=0)

    def solve_nonnegative(self, demands: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Non-negative least squares (scipy.optimize.nnls) for each day.

        Returns (solution size x n_days, residual norm per day in kWh). The
        residual is 0 on feasible days and measures the demand that cannot be
        met exactly on infeasible ones.
        """
        D = self._check_demands(demands)
        D2 = D.reshape(self.size, -1)
        sols, res = [], []
        for j in range(D2.shape[1]):
            s, r = scipy.optimize.nnls(self._A, D2[:, j])
            sols.append(s)
            res.append(r)
        return np.column_stack(sols), np.array(res)

    def clip_negative(self, solution: ArrayLike) -> np.ndarray:
        """Alternative strategy: set negative energies to zero (breaks the equations)."""
        return np.clip(np.asarray(solution, dtype=float), 0.0, None)

    # ----- cost -------------------------------------------------------------------
    def daily_cost(self, solution: ArrayLike) -> np.ndarray:
        """Cost per day in UGX: sum over sources of energy x unit cost."""
        S = np.asarray(solution, dtype=float)
        if S.shape[0] != self.size:
            raise ValueError(f"solution must have {self.size} rows")
        return self.costs @ S


class HybridMicroGrid(MicroGrid):
    """Micro-grid with a diesel generator z and a third constraint (3 x 3 system).

    Default constraints (diesel assumed to contribute one-for-one to both
    original loads, plus an evening/night load that solar cannot serve):

        3x + 2y + z = D1   (daytime load)
        4x +  y + z = D2   (critical-equipment load)
             y + z = D3   (night load)
    """

    def __init__(
        self,
        coefficients: ArrayLike = ((3.0, 2.0, 1.0), (4.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
        sources: Sequence[str] = ("solar", "battery", "diesel"),
        costs: Sequence[float] = (150.0, 450.0, 1200.0),
    ) -> None:
        super().__init__(coefficients, sources, costs)
        if self.size != 3:
            raise ValueError("HybridMicroGrid needs a 3 x 3 system")

    def dependent_rows(self) -> bool:
        """True if one constraint is a linear combination of the others (rank < 3)."""
        return self.rank() < self.size


# ----- data -------------------------------------------------------------------------
def generate_demand(
    n_days: int = 30,
    *,
    rng: np.random.Generator,
    base_day: float = 100.0,
    critical_ratio: float = 1.2,
    weekend_factor: float = 0.85,
    noise_sd: float = 4.0,
) -> pd.DataFrame:
    """Synthetic daily demand (kWh) with a weekly pattern; day 1 = Monday.

    D1 = base_day x (weekend_factor on Sat/Sun) + noise;
    D2 = critical_ratio x D1 + independent noise. Values are rounded to 0.1 kWh.
    """
    if n_days < 1 or base_day <= 0 or noise_sd < 0:
        raise ValueError("n_days >= 1, base_day > 0 and noise_sd >= 0 are required")
    weekday = np.arange(n_days) % 7
    weekly = np.where(weekday >= 5, weekend_factor, 1.0)
    d1 = base_day * weekly + rng.normal(0, noise_sd, n_days)
    d2 = critical_ratio * d1 + rng.normal(0, noise_sd, n_days)
    return pd.DataFrame({
        "day": np.arange(1, n_days + 1),
        "weekday": np.array(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])[weekday],
        "D1_kWh": np.round(np.clip(d1, 0, None), 1),
        "D2_kWh": np.round(np.clip(d2, 0, None), 1),
    })


def save_demand_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_demand_csv(path: str | Path) -> np.ndarray:
    """Load D1, D2 columns as a 2 x n_days array, validating the values."""
    df = pd.read_csv(path)
    missing = {"D1_kWh", "D2_kWh"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    D = df[["D1_kWh", "D2_kWh"]].to_numpy(dtype=float).T
    if D.shape[1] == 0 or not np.all(np.isfinite(D)) or np.any(D < 0):
        raise ValueError("CSV demands must be non-empty, finite and non-negative")
    return D


# ----- interactive input --------------------------------------------------------------
def parse_demand(text: str) -> float:
    """Convert one typed value to a demand in kWh, or raise ValueError with a reason."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty input")
    try:
        value = float(cleaned)
    except ValueError:
        raise ValueError(f"{cleaned!r} is not a number") from None
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    if value < 0:
        raise ValueError("demand cannot be negative")
    return value


def read_demands_interactive(
    labels: Sequence[str] = ("D1 daytime load (kWh)", "D2 critical load (kWh)"),
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    max_attempts: int = 10,
) -> list[float]:
    """Prompt for each demand, re-prompting on empty, non-numeric or negative input.

    `input_fn` and `print_fn` can be replaced (e.g. in tests or a scripted
    notebook demo) so the function never blocks an automated run.
    """
    values = []
    for label in labels:
        for _ in range(max_attempts):
            try:
                values.append(parse_demand(input_fn(f"{label}: ")))
                break
            except ValueError as err:
                print_fn(f"  Invalid: {err}. Please try again.")
        else:
            raise ValueError(f"no valid value for {label} after {max_attempts} attempts")
    return values
