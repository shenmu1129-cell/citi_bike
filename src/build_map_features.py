from __future__ import annotations

from io import StringIO
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import requests

from .regional_common import PATHS, ensure_dirs, haversine_m, log


MTA_STATIONS_URL = "https://data.ny.gov/resource/i9wp-a4ja.csv?$limit=5000"
MAP_FEATURE_COLUMNS = [
    "nearest_subway_distance",
    "subway_count_500m",
    "subway_count_1000m",
    "transit_congestion_index",
]


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _load_subway_raw() -> pd.DataFrame:
    ensure_dirs()
    local_path = PATHS["raw_spatial"] / "mta_subway_stations.csv"
    if local_path.exists():
        log(f"use local subway station file: {local_path}")
        return pd.read_csv(local_path)
    try:
        log("download MTA subway stations")
        response = requests.get(MTA_STATIONS_URL, timeout=20)
        response.raise_for_status()
        raw = pd.read_csv(StringIO(response.text))
        raw.to_csv(local_path, index=False)
        return raw
    except Exception as exc:
        log(f"MTA subway station download failed; continue with empty map features: {exc}")
        return pd.DataFrame()


def _normalise_subway_stations(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["station_key", "station_name", "lat", "lng"])
    lat_col = _first_existing(
        raw.columns,
        [
            "gtfs_latitude",
            "latitude",
            "entrance_latitude",
            "stop_lat",
            "lat",
            "station_latitude",
        ],
    )
    lng_col = _first_existing(
        raw.columns,
        [
            "gtfs_longitude",
            "longitude",
            "entrance_longitude",
            "stop_lon",
            "lon",
            "lng",
            "station_longitude",
        ],
    )
    if lat_col is None or lng_col is None:
        log("MTA station file has no recognizable latitude/longitude columns")
        return pd.DataFrame(columns=["station_key", "station_name", "lat", "lng"])

    key_col = _first_existing(raw.columns, ["gtfs_stop_id", "station_id", "complex_id", "stop_id", "name"])
    name_col = _first_existing(raw.columns, ["stop_name", "station_name", "name", "line"])
    stations = pd.DataFrame(
        {
            "station_key": raw[key_col].astype(str) if key_col else raw.index.astype(str),
            "station_name": raw[name_col].astype(str) if name_col else "",
            "lat": pd.to_numeric(raw[lat_col], errors="coerce"),
            "lng": pd.to_numeric(raw[lng_col], errors="coerce"),
        }
    )
    stations = stations.dropna(subset=["lat", "lng"])
    stations = stations[stations["lat"].between(40.45, 41.05) & stations["lng"].between(-74.35, -73.55)]
    if stations.empty:
        return pd.DataFrame(columns=["station_key", "station_name", "lat", "lng"])
    stations = (
        stations.groupby("station_key", as_index=False)
        .agg(station_name=("station_name", "first"), lat=("lat", "mean"), lng=("lng", "mean"))
        .sort_values(["lat", "lng"])
        .reset_index(drop=True)
    )
    return stations


def compute_map_features(config: Dict) -> pd.DataFrame:
    ensure_dirs()
    region_path = PATHS["tables"] / "region_grid_info.csv"
    if not region_path.exists():
        raise FileNotFoundError("region_grid_info.csv is required before building map features")
    regions = pd.read_csv(region_path)
    raw = _load_subway_raw()
    stations = _normalise_subway_stations(raw)
    stations.to_csv(PATHS["tables"] / "subway_station_points.csv", index=False)

    features = regions[["grid_id", "grid_center_lat", "grid_center_lng"]].copy()
    if stations.empty:
        features["nearest_subway_distance"] = 9999.0
        features["subway_count_500m"] = 0
        features["subway_count_1000m"] = 0
    else:
        station_lat = stations["lat"].to_numpy()
        station_lng = stations["lng"].to_numpy()
        nearest = []
        count_500 = []
        count_1000 = []
        for row in features.itertuples(index=False):
            distances = haversine_m(row.grid_center_lat, row.grid_center_lng, station_lat, station_lng)
            nearest.append(float(np.min(distances)))
            count_500.append(int(np.sum(distances <= 500)))
            count_1000.append(int(np.sum(distances <= 1000)))
        features["nearest_subway_distance"] = nearest
        features["subway_count_500m"] = count_500
        features["subway_count_1000m"] = count_1000

    features["transit_congestion_index"] = (
        features["subway_count_500m"].astype(float) * 1.5
        + features["subway_count_1000m"].astype(float) * 0.5
        + 1000.0 / features["nearest_subway_distance"].clip(lower=100.0)
    )
    features.to_csv(PATHS["tables"] / "region_map_features.csv", index=False)
    log(f"saved subway map features: {features.shape}, stations={len(stations)}")
    return features
