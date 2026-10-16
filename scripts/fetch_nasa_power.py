"""Download 10 years (2014-2023) of monthly rainfall from NASA POWER for the MP4 regions.

Run once from the repository root (needs an internet connection):

    python scripts/fetch_nasa_power.py

Raw JSON responses go to data/raw/rainfall/ and monthly totals (mm) to
data/processed/rainfall_nasa_power.csv, which the MP4 notebook reads.
"""

import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from a1.rainfall import parse_power_monthly

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "rainfall"
OUT = ROOT / "data" / "processed" / "rainfall_nasa_power.csv"

# Approximate town-centre coordinates (latitude, longitude).
REGIONS = {"Kampala": (0.3476, 32.5825), "Gulu": (2.7746, 32.2990), "Mbarara": (-0.6072, 30.6545)}
START, END = 2014, 2023
URL = ("https://power.larc.nasa.gov/api/temporal/monthly/point?parameters=PRECTOTCORR"
       "&community=AG&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON")


def fetch(lat: float, lon: float) -> dict:
    url = URL.format(lat=lat, lon=lon, start=START, end=END)
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, (lat, lon) in REGIONS.items():
        print(f"Downloading {name} ({lat}, {lon}) ...")
        payload = fetch(lat, lon)
        (RAW / f"nasa_power_{name.lower()}_{START}_{END}.json").write_text(json.dumps(payload, indent=1))
        df = parse_power_monthly(payload)
        df.insert(0, "region", name)
        frames.append(df)
        time.sleep(1)                                   # be polite to the API
    data = pd.concat(frames, ignore_index=True)
    expected = len(REGIONS) * (END - START + 1) * 12
    if len(data) != expected:
        raise ValueError(f"expected {expected} monthly rows, got {len(data)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT, index=False)
    print(f"Wrote {len(data)} rows to {OUT.relative_to(ROOT)}")
    print(data.groupby("region")["rain_mm"].sum().div(END - START + 1).round(0)
          .rename("mean annual rainfall (mm)").to_string())


if __name__ == "__main__":
    main()
