#!/usr/bin/env python3
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def show_path(path: Path, label: str) -> None:
    if path.exists():
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"- {label}: {path.relative_to(ROOT)} ({size_mb:.2f} MB)")
    else:
        print(f"- {label}: {path.relative_to(ROOT)} [missing]")


def show_csv(path: Path, label: str, n: int = 3) -> None:
    print(f"\n[{label}] {path.relative_to(ROOT)}")
    if not path.exists():
        print("  missing")
        return
    df = pd.read_csv(path)
    print(f"  shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  columns: {', '.join(df.columns)}")
    print(df.head(n).to_string(index=False))


def main() -> None:
    raw_citibike = ROOT / "data" / "raw" / "citibike"
    raw_weather = ROOT / "data" / "raw" / "weather" / "weather_raw.json"
    hourly_rides = ROOT / "data" / "interim" / "hourly_rides.csv"
    hourly_weather = ROOT / "data" / "interim" / "hourly_weather.csv"
    model_dataset = ROOT / "data" / "processed" / "model_dataset.csv"

    print("Citi Bike demand prediction data pipeline\n")
    print("1) Raw input files")
    for zip_path in sorted(raw_citibike.glob("*.zip")):
        show_path(zip_path, "Citi Bike trip zip")
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            print(f"  contains: {', '.join(names[:3])}")
    show_path(raw_weather, "Open-Meteo raw API response")

    print("\n2) Weather API connection used in run_all.py")
    print("  endpoint: https://archive-api.open-meteo.com/v1/archive")
    print("  latitude/longitude: 40.7128, -74.0060")
    print("  timezone: America/New_York")
    print("  hourly variables:")
    print("    temperature_2m, relative_humidity_2m, precipitation, rain, wind_speed_10m, cloud_cover")
    if raw_weather.exists():
        raw = json.loads(raw_weather.read_text(encoding="utf-8"))
        hourly = raw.get("hourly", {})
        times = hourly.get("time", [])
        if times:
            print(f"  saved response covers: {times[0]} to {times[-1]}")

    print("\n3) Generated analysis tables")
    show_csv(hourly_rides, "Hourly Citi Bike demand")
    show_csv(hourly_weather, "Hourly weather")
    show_csv(model_dataset, "Final merged model dataset", n=2)

    print("\n4) Main code locations for presentation")
    print("  Citi Bike download/read/aggregation: run_all.py lines 79-186")
    print("  Open-Meteo API request: run_all.py lines 188-221")
    print("  Weather preprocessing: run_all.py lines 223-235")
    print("  Ride + weather feature merge: run_all.py lines 239-280")
    print("  Modeling and evaluation: run_all.py lines 547-610")


if __name__ == "__main__":
    main()
