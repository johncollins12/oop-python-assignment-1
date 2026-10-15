"""Build data/processed/population.csv for Mini-Project 1.

Combines
  * the three illustrative districts given in the assignment brief, and
  * real UBOS figures for Jinja and Busia:
      2015-2021  UBOS mid-year projections (raw file 1)
      2024       NPHC 2024 census counts (raw file 2); Jinja = district + city
      2022-2023  geometric interpolation between 2021 and 2024 (labelled)

Every row carries a `source` label. Run from the repository root:

    python scripts/build_population_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from a1.population import geometric_interpolate

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "population"
OUT = ROOT / "data" / "processed" / "population.csv"

PROJECTIONS_FILE = RAW / (
    "Census_Population_counts_(2002_and_2014)_by_Region,_District_and_"
    "Mid-Year_Population_projections_(2015-2021).xlsx"
)
CENSUS_FILE = RAW / "NPHC-2024-Subcounty-Profiles-Excel-Tables.xlsx"

YEARS = list(range(2015, 2025))
PROJECTION_YEARS = list(range(2015, 2022))

# Illustrative data from the assignment brief (thousands), 2015-2024.
BRIEF = {
    "Kampala": [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800],
    "Wakiso": [950, 1000, 1070, 1150, 1220, 1300, 1390, 1480, 1570, 1670],
    "Gulu": [320, 330, 345, 360, 375, 390, 410, 430, 455, 480],
}
SRC_BRIEF = "illustrative (assignment brief)"
SRC_PROJ = "UBOS mid-year projection"
SRC_INTERP = "geometric interpolation 2021-2024"
SRC_CENSUS = "NPHC 2024 census"

# District -> census units summed for 2024 (Jinja City was split off in 2020).
CENSUS_UNITS = {"Jinja": ["JINJA", "JINJA CITY"], "Busia": ["BUSIA"]}


def read_projections(district: str) -> list[float]:
    """Return 2015-2021 projected populations (persons) for one district."""
    sheet = pd.read_excel(PROJECTIONS_FILE, sheet_name="Sheet3", header=None)
    header_rows = sheet.index[sheet.eq(2015).any(axis=1)]
    if len(header_rows) != 1:
        raise ValueError("could not find a unique year-header row in the projections file")
    header = sheet.loc[header_rows[0]]
    year_cols = {int(v): c for c, v in header.items() if v in PROJECTION_YEARS}
    name_col = sheet.columns[sheet.eq("Region/District").any(axis=0)][0]
    names = sheet[name_col].astype(str).str.strip().str.casefold()
    rows = sheet.index[names == district.casefold()]
    if len(rows) != 1:
        raise ValueError(f"expected one row for {district!r}, found {len(rows)}")
    return [float(sheet.at[rows[0], year_cols[y]]) for y in PROJECTION_YEARS]


def read_census_total(unit: str, table: pd.DataFrame) -> float:
    """Return the 2024 total (persons) for one census unit name.

    Some names repeat at lower levels (e.g. a BUSIA subcounty inside BUSIA
    district). The table lists each district before its sub-units, so the
    first match is the district; we check it is also the largest.
    """
    names = table[0].astype(str).str.strip().str.upper()
    totals = table.loc[names == unit, 3].astype(float)
    if totals.empty:
        raise ValueError(f"census unit {unit!r} not found")
    first = float(totals.iloc[0])
    if first != totals.max():
        raise ValueError(f"first match for {unit!r} is not the largest; check the table")
    return first


def build() -> pd.DataFrame:
    rows = []
    for district, values in BRIEF.items():
        rows += [(district, y, float(v), SRC_BRIEF) for y, v in zip(YEARS, values)]

    census = pd.read_excel(CENSUS_FILE, sheet_name="Table1", header=None)
    for district, units in CENSUS_UNITS.items():
        projected = read_projections(district)
        total_2024 = sum(read_census_total(u, census) for u in units)
        bridge = geometric_interpolate(projected[-1], total_2024, 3)  # 2021..2024
        persons = projected + [bridge[1], bridge[2], total_2024]
        sources = [SRC_PROJ] * 7 + [SRC_INTERP] * 2 + [SRC_CENSUS]
        rows += [
            (district, y, round(p / 1000.0, 3), s)
            for y, p, s in zip(YEARS, persons, sources)
        ]

    df = pd.DataFrame(rows, columns=["district", "year", "population_thousands", "source"])
    if df.groupby("district")["year"].apply(list).map(lambda ys: ys != YEARS).any():
        raise ValueError("every district must have exactly the years 2015-2024")
    if not np.all(df["population_thousands"] > 0):
        raise ValueError("all populations must be positive")
    return df


def main() -> None:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT.relative_to(ROOT)}")
    print(df[df["district"].isin(CENSUS_UNITS)].to_string(index=False))


if __name__ == "__main__":
    main()