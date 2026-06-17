#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CONFIG = {
    "months": ["202603", "202604", "202605"],
    "citibike_url_template": "https://s3.amazonaws.com/tripdata/{month}-citibike-tripdata.zip",
    "grid_size": 0.01,
    "test_size": 0.2,
    "random_state": 42,
    "quick": False,
    "max_train_rows": 600_000,
    "max_poi_grids": 900,
    "subway_urls": [
        "https://data.ny.gov/resource/i9wp-a4ja.csv?$limit=5000",
        "https://data.ny.gov/api/views/i9wp-a4ja/rows.csv?accessType=DOWNLOAD",
    ],
    "gbfs_station_information_urls": [
        "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_information.json",
        "https://gbfs.citibikenyc.com/gbfs/en/station_information.json",
    ],
    "gbfs_station_status_urls": [
        "https://gbfs.lyft.com/gbfs/2.3/bkn/en/station_status.json",
        "https://gbfs.citibikenyc.com/gbfs/en/station_status.json",
    ],
    "overpass_urls": [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ],
}


@dataclass
class Paths:
    root: Path = ROOT
    raw_citibike: Path = ROOT / "data" / "raw" / "citibike"
    raw_spatial: Path = ROOT / "data" / "raw" / "spatial"
    interim: Path = ROOT / "data" / "interim"
    processed: Path = ROOT / "data" / "processed"
    figures: Path = ROOT / "outputs" / "figures"
    tables: Path = ROOT / "outputs" / "tables"
    models: Path = ROOT / "outputs" / "models"
    cache: Path = ROOT / ".cache" / "spatial"

    def ensure(self) -> None:
        for p in [
            self.raw_citibike,
            self.raw_spatial,
            self.interim,
            self.processed,
            self.figures,
            self.tables,
            self.models,
            self.cache,
            ROOT / ".cache" / "matplotlib",
        ]:
            p.mkdir(parents=True, exist_ok=True)


P = Paths()
sns.set_theme(style="whitegrid", font="Arial Unicode MS")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(P.figures / name, bbox_inches="tight")
    plt.close()


def download_file(url: str, out: Path, timeout: int = 60) -> None:
    if out.exists() and out.stat().st_size > 10_000_000:
        log(f"skip existing {out.name} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
        return
    tmp = out.with_suffix(out.suffix + ".part")
    log(f"download {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(out)


def download_citibike(months: List[str]) -> List[Path]:
    zips = []
    for month in months:
        out = P.raw_citibike / f"{month}-citibike-tripdata.zip"
        download_file(CONFIG["citibike_url_template"].format(month=month), out)
        zips.append(out)
    return zips


def read_zip_csvs(zip_path: Path, quick: bool = False) -> Iterable[pd.DataFrame]:
    usecols = {
        "ride_id",
        "rideable_type",
        "started_at",
        "ended_at",
        "start_lat",
        "start_lng",
        "end_lat",
        "end_lng",
    }
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No CSV found inside {zip_path}")
        for name in names:
            log(f"read {zip_path.name}/{name}")
            with zf.open(name) as f:
                reader = pd.read_csv(
                    f,
                    usecols=lambda c: c in usecols,
                    chunksize=300_000,
                    low_memory=False,
                )
                for i, chunk in enumerate(reader):
                    yield chunk
                    if quick and i >= 1:
                        log("quick mode: stop after two chunks per csv")
                        break


def grid_values(lat: pd.Series, lng: pd.Series, grid_size: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    grid_lat = (pd.to_numeric(lat, errors="coerce") / grid_size).round() * grid_size
    grid_lng = (pd.to_numeric(lng, errors="coerce") / grid_size).round() * grid_size
    grid_id = grid_lat.map(lambda x: f"{x:.2f}") + "_" + grid_lng.map(lambda x: f"{x:.2f}")
    return grid_lat, grid_lng, grid_id


def valid_nyc_coords(df: pd.DataFrame, lat_col: str, lng_col: str) -> pd.Series:
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lng = pd.to_numeric(df[lng_col], errors="coerce")
    return lat.between(40.45, 41.05) & lng.between(-74.35, -73.55)


def build_regional_flow(zips: List[Path], quick: bool = False) -> pd.DataFrame:
    pickup_records = []
    dropoff_records = []
    total_rows = 0
    for z in zips:
        for chunk in read_zip_csvs(z, quick=quick):
            total_rows += len(chunk)
            chunk = chunk.dropna(subset=["ride_id", "started_at", "ended_at"]).drop_duplicates("ride_id")
            chunk["started_at"] = pd.to_datetime(chunk["started_at"], errors="coerce")
            chunk["ended_at"] = pd.to_datetime(chunk["ended_at"], errors="coerce")
            chunk = chunk.dropna(subset=["started_at", "ended_at"])
            duration = (chunk["ended_at"] - chunk["started_at"]).dt.total_seconds() / 60
            chunk = chunk[(duration >= 1) & (duration <= 1440)].copy()

            start_valid = valid_nyc_coords(chunk, "start_lat", "start_lng")
            if start_valid.any():
                start = chunk.loc[start_valid, ["ride_id", "started_at", "start_lat", "start_lng"]].copy()
                start["datetime_hour"] = start["started_at"].dt.floor("h")
                start["grid_lat"], start["grid_lng"], start["grid_id"] = grid_values(
                    start["start_lat"], start["start_lng"], CONFIG["grid_size"]
                )
                pickup_records.append(
                    start.groupby(["datetime_hour", "grid_id"], as_index=False)
                    .agg(pickup_count=("ride_id", "count"))
                )

            end_valid = valid_nyc_coords(chunk, "end_lat", "end_lng")
            if end_valid.any():
                end = chunk.loc[end_valid, ["ride_id", "ended_at", "end_lat", "end_lng"]].copy()
                end["datetime_hour"] = end["ended_at"].dt.floor("h")
                end["grid_lat"], end["grid_lng"], end["grid_id"] = grid_values(
                    end["end_lat"], end["end_lng"], CONFIG["grid_size"]
                )
                dropoff_records.append(
                    end.groupby(["datetime_hour", "grid_id"], as_index=False)
                    .agg(dropoff_count=("ride_id", "count"))
                )

    pickups = pd.concat(pickup_records, ignore_index=True).groupby(["datetime_hour", "grid_id"], as_index=False).sum()
    dropoffs = pd.concat(dropoff_records, ignore_index=True).groupby(["datetime_hour", "grid_id"], as_index=False).sum()
    active_grids = pd.Index(sorted(set(pickups["grid_id"]).union(set(dropoffs["grid_id"]))))
    hours = pd.date_range(
        min(pickups["datetime_hour"].min(), dropoffs["datetime_hour"].min()),
        max(pickups["datetime_hour"].max(), dropoffs["datetime_hour"].max()),
        freq="h",
    )
    panel = pd.MultiIndex.from_product([hours, active_grids], names=["datetime_hour", "grid_id"]).to_frame(index=False)
    flow = panel.merge(pickups, on=["datetime_hour", "grid_id"], how="left").merge(
        dropoffs, on=["datetime_hour", "grid_id"], how="left"
    )
    flow[["pickup_count", "dropoff_count"]] = flow[["pickup_count", "dropoff_count"]].fillna(0).astype("int32")
    flow["net_flow"] = flow["dropoff_count"] - flow["pickup_count"]
    centers = parse_grid_centers(active_grids.to_series(index=active_grids).reset_index(drop=True))
    flow = flow.merge(centers, on="grid_id", how="left")
    flow.to_csv(P.interim / "regional_hourly_flow.csv", index=False)
    log(f"regional hourly flow: {flow.shape}, raw rows read: {total_rows}")
    return flow


def parse_grid_centers(grid_ids: pd.Series) -> pd.DataFrame:
    parts = grid_ids.astype(str).str.split("_", expand=True)
    return pd.DataFrame(
        {
            "grid_id": grid_ids.to_numpy(),
            "grid_lat": parts[0].astype(float).to_numpy(),
            "grid_lng": parts[1].astype(float).to_numpy(),
        }
    )


def haversine_m(lat1, lng1, lat2, lng2):
    r = 6_371_000.0
    lat1 = np.radians(lat1)
    lng1 = np.radians(lng1)
    lat2 = np.radians(lat2)
    lng2 = np.radians(lng2)
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def request_csv_candidates(urls: List[str], local_name: str) -> Optional[pd.DataFrame]:
    local_path = P.raw_spatial / local_name
    if local_path.exists():
        try:
            return pd.read_csv(local_path)
        except Exception as exc:
            log(f"local {local_name} unreadable: {exc}")
    for url in urls:
        try:
            log(f"download spatial csv: {url}")
            df = pd.read_csv(url)
            if not df.empty:
                df.to_csv(local_path, index=False)
                return df
        except Exception as exc:
            log(f"spatial csv failed: {exc}")
    return None


def normalize_subway_stations(raw: Optional[pd.DataFrame]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["station_name", "lat", "lng"])
    lower = {c.lower().strip().replace(" ", "_"): c for c in raw.columns}
    lat_col = next((lower[c] for c in ["gtfs_latitude", "entrance_latitude", "latitude", "lat"] if c in lower), None)
    lng_col = next((lower[c] for c in ["gtfs_longitude", "entrance_longitude", "longitude", "lon", "lng"] if c in lower), None)
    name_col = next((lower[c] for c in ["stop_name", "station_name", "constituent_station_name", "name"] if c in lower), None)
    if not lat_col or not lng_col:
        return pd.DataFrame(columns=["station_name", "lat", "lng"])
    station_key_col = next((lower[c] for c in ["gtfs_stop_id", "station_id", "complex_id"] if c in lower), None)
    out = pd.DataFrame(
        {
            "station_key": raw[station_key_col].astype(str) if station_key_col else raw.index.astype(str),
            "station_name": raw[name_col].astype(str) if name_col else "",
            "lat": pd.to_numeric(raw[lat_col], errors="coerce"),
            "lng": pd.to_numeric(raw[lng_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["lat", "lng"])
    out = out[out["lat"].between(40.45, 41.05) & out["lng"].between(-74.35, -73.55)]
    out = (
        out.groupby("station_key", as_index=False)
        .agg(station_name=("station_name", "first"), lat=("lat", "mean"), lng=("lng", "mean"))
        .drop_duplicates(["station_name", "lat", "lng"])
        .reset_index(drop=True)
    )
    return out


def compute_subway_features(grids: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw = request_csv_candidates(CONFIG["subway_urls"], "mta_subway_stations.csv")
    stations = normalize_subway_stations(raw)
    if stations.empty:
        log("subway station data unavailable; using offline zero-distance fallback")
        features = grids[["grid_id"]].copy()
        features["nearest_subway_distance"] = np.nan
        features["subway_count_500m"] = 0
        features["subway_count_1000m"] = 0
        return features, stations

    rows = []
    st_lat = stations["lat"].to_numpy()
    st_lng = stations["lng"].to_numpy()
    for row in grids.itertuples(index=False):
        d = haversine_m(row.grid_lat, row.grid_lng, st_lat, st_lng)
        rows.append(
            {
                "grid_id": row.grid_id,
                "nearest_subway_distance": float(np.min(d)),
                "subway_count_500m": int(np.sum(d <= 500)),
                "subway_count_1000m": int(np.sum(d <= 1000)),
            }
        )
    return pd.DataFrame(rows), stations


def fetch_json_candidates(urls: List[str], local_name: str) -> Optional[dict]:
    local_path = P.raw_spatial / local_name
    if local_path.exists():
        try:
            return json.loads(local_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log(f"local {local_name} unreadable: {exc}")
    for url in urls:
        try:
            log(f"download json: {url}")
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            local_path.write_text(json.dumps(data), encoding="utf-8")
            return data
        except Exception as exc:
            log(f"json download failed: {exc}")
    return None


def gbfs_records(data: Optional[dict], key: str) -> List[dict]:
    if not data:
        return []
    payload = data.get("data", data)
    records = payload.get(key, [])
    return records if isinstance(records, list) else []


def compute_gbfs_features(grids: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    info = fetch_json_candidates(CONFIG["gbfs_station_information_urls"], "gbfs_station_information.json")
    status = fetch_json_candidates(CONFIG["gbfs_station_status_urls"], "gbfs_station_status.json")
    info_df = pd.DataFrame(gbfs_records(info, "stations"))
    status_df = pd.DataFrame(gbfs_records(status, "stations"))
    if info_df.empty:
        log("GBFS station_information unavailable; using zero station status fallback")
        features = grids[["grid_id"]].copy()
        for col in ["station_count", "total_capacity", "current_bikes", "current_empty_docks", "current_ebikes"]:
            features[col] = 0
        return features, pd.DataFrame(columns=["station_id", "lat", "lng"])

    keep = ["station_id", "name", "lat", "lon", "capacity"]
    for col in keep:
        if col not in info_df:
            info_df[col] = np.nan
    station_df = info_df[keep].copy()
    station_df["lat"] = pd.to_numeric(station_df["lat"], errors="coerce")
    station_df["lng"] = pd.to_numeric(station_df["lon"], errors="coerce")
    station_df["capacity"] = pd.to_numeric(station_df["capacity"], errors="coerce").fillna(0)
    station_df = station_df.dropna(subset=["station_id", "lat", "lng"])
    if not status_df.empty and "station_id" in status_df.columns:
        status_keep = ["station_id", "num_bikes_available", "num_docks_available", "num_ebikes_available"]
        for col in status_keep:
            if col not in status_df:
                status_df[col] = 0
        station_df = station_df.merge(status_df[status_keep], on="station_id", how="left")
    for col in ["num_bikes_available", "num_docks_available", "num_ebikes_available"]:
        if col not in station_df:
            station_df[col] = 0
        station_df[col] = pd.to_numeric(station_df[col], errors="coerce").fillna(0)
    station_df["grid_lat"], station_df["grid_lng"], station_df["grid_id"] = grid_values(
        station_df["lat"], station_df["lng"], CONFIG["grid_size"]
    )
    features = station_df.groupby("grid_id", as_index=False).agg(
        station_count=("station_id", "nunique"),
        total_capacity=("capacity", "sum"),
        current_bikes=("num_bikes_available", "sum"),
        current_empty_docks=("num_docks_available", "sum"),
        current_ebikes=("num_ebikes_available", "sum"),
    )
    out = grids[["grid_id"]].merge(features, on="grid_id", how="left")
    for col in ["station_count", "total_capacity", "current_bikes", "current_empty_docks", "current_ebikes"]:
        out[col] = out[col].fillna(0)
    return out, station_df


def overpass_query(bbox: Tuple[float, float, float, float]) -> str:
    south, west, north, east = bbox
    return f"""
[out:json][timeout:90];
(
  node["shop"]({south},{west},{north},{east});
  way["shop"]({south},{west},{north},{east});
  relation["shop"]({south},{west},{north},{east});
  node["amenity"="restaurant"]({south},{west},{north},{east});
  way["amenity"="restaurant"]({south},{west},{north},{east});
  relation["amenity"="restaurant"]({south},{west},{north},{east});
  node["amenity"="cafe"]({south},{west},{north},{east});
  way["amenity"="cafe"]({south},{west},{north},{east});
  relation["amenity"="cafe"]({south},{west},{north},{east});
  node["amenity"="school"]({south},{west},{north},{east});
  way["amenity"="school"]({south},{west},{north},{east});
  relation["amenity"="school"]({south},{west},{north},{east});
  node["leisure"="park"]({south},{west},{north},{east});
  way["leisure"="park"]({south},{west},{north},{east});
  relation["leisure"="park"]({south},{west},{north},{east});
  node["landuse"="residential"]({south},{west},{north},{east});
  way["landuse"="residential"]({south},{west},{north},{east});
  relation["landuse"="residential"]({south},{west},{north},{east});
  node["landuse"="commercial"]({south},{west},{north},{east});
  way["landuse"="commercial"]({south},{west},{north},{east});
  relation["landuse"="commercial"]({south},{west},{north},{east});
);
out center;
"""


def fetch_overpass_pois(grids: pd.DataFrame) -> pd.DataFrame:
    cache_path = P.cache / "overpass_pois.json"
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            data = None
    else:
        data = None
    if data is None:
        south = max(40.45, float(grids["grid_lat"].min()) - 0.02)
        north = min(41.05, float(grids["grid_lat"].max()) + 0.02)
        west = max(-74.35, float(grids["grid_lng"].min()) - 0.02)
        east = min(-73.55, float(grids["grid_lng"].max()) + 0.02)
        query = overpass_query((south, west, north, east))
        last_error = None
        for url in CONFIG["overpass_urls"]:
            try:
                log(f"query Overpass API for POIs: {url}")
                r = requests.post(
                    url,
                    data=query.encode("utf-8"),
                    headers={"Content-Type": "text/plain; charset=utf-8", "User-Agent": "citibike-regional-spatial-analysis"},
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()
                cache_path.write_text(json.dumps(data), encoding="utf-8")
                last_error = None
                break
            except Exception as exc:
                detail = getattr(getattr(exc, "response", None), "text", "")
                last_error = f"{exc} {detail[:200]}".strip()
                log(f"Overpass endpoint failed: {last_error}")
        if last_error is not None:
            log(f"Overpass unavailable; POI features use offline zero fallback: {last_error}")
            return pd.DataFrame(columns=["lat", "lng", "poi_type"])

    rows = []
    for el in data.get("elements", []):
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lng = el.get("lon") or el.get("center", {}).get("lon")
        tags = el.get("tags", {})
        if lat is None or lng is None:
            continue
        poi_type = None
        if tags.get("shop"):
            poi_type = "shop"
        elif tags.get("amenity") in {"restaurant", "cafe", "school"}:
            poi_type = tags.get("amenity")
        elif tags.get("leisure") == "park":
            poi_type = "park"
        elif tags.get("landuse") in {"residential", "commercial"}:
            poi_type = tags.get("landuse")
        if poi_type:
            rows.append({"lat": float(lat), "lng": float(lng), "poi_type": poi_type})
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["lat", "lng", "poi_type"])


def compute_poi_features(grids: pd.DataFrame, quick: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    poi_types = ["shop", "restaurant", "cafe", "school", "park", "residential", "commercial"]
    pois = fetch_overpass_pois(grids)
    features = grids[["grid_id", "grid_lat", "grid_lng"]].copy()
    for poi_type in poi_types:
        features[f"{poi_type}_count_500m"] = 0
    if pois.empty:
        return features.drop(columns=["grid_lat", "grid_lng"]), pois

    scored_grids = features
    if quick and len(scored_grids) > CONFIG["max_poi_grids"]:
        scored_grids = scored_grids.head(CONFIG["max_poi_grids"]).copy()
    poi_lat = pois["lat"].to_numpy()
    poi_lng = pois["lng"].to_numpy()
    poi_type_arr = pois["poi_type"].to_numpy()
    rows = []
    for row in scored_grids.itertuples(index=False):
        d = haversine_m(row.grid_lat, row.grid_lng, poi_lat, poi_lng)
        nearby = d <= 500
        item = {"grid_id": row.grid_id}
        for poi_type in poi_types:
            item[f"{poi_type}_count_500m"] = int(np.sum(nearby & (poi_type_arr == poi_type)))
        rows.append(item)
    counted = pd.DataFrame(rows)
    out = grids[["grid_id"]].merge(counted, on="grid_id", how="left")
    for poi_type in poi_types:
        out[f"{poi_type}_count_500m"] = out[f"{poi_type}_count_500m"].fillna(0).astype(int)
    return out, pois


def build_model_dataset(flow: pd.DataFrame, quick: bool = False) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    grids = flow[["grid_id", "grid_lat", "grid_lng"]].drop_duplicates().reset_index(drop=True)
    subway_features, subway_stations = compute_subway_features(grids)
    gbfs_features, gbfs_stations = compute_gbfs_features(grids)
    poi_features, pois = compute_poi_features(grids, quick=quick)
    spatial = (
        grids.merge(subway_features, on="grid_id", how="left")
        .merge(poi_features, on="grid_id", how="left")
        .merge(gbfs_features, on="grid_id", how="left")
    )
    numeric_spatial_cols = [c for c in spatial.columns if c not in {"grid_id"}]
    spatial[numeric_spatial_cols] = spatial[numeric_spatial_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    df = flow.merge(spatial, on=["grid_id", "grid_lat", "grid_lng"], how="left")
    df = df.sort_values(["grid_id", "datetime_hour"]).reset_index(drop=True)
    df["hour"] = df["datetime_hour"].dt.hour
    df["weekday"] = df["datetime_hour"].dt.weekday
    df["month"] = df["datetime_hour"].dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    for lag in [1, 2, 3, 24, 168]:
        df[f"net_flow_lag_{lag}"] = df.groupby("grid_id")["net_flow"].shift(lag)
        df[f"pickup_lag_{lag}"] = df.groupby("grid_id")["pickup_count"].shift(lag)
        df[f"dropoff_lag_{lag}"] = df.groupby("grid_id")["dropoff_count"].shift(lag)
    for win in [3, 6, 24]:
        df[f"net_flow_rolling_mean_{win}"] = df.groupby("grid_id")["net_flow"].transform(
            lambda s: s.shift(1).rolling(win).mean()
        )
    df["historical_avg_demand"] = (
        df.groupby("grid_id")["pickup_count"].transform(lambda s: s.shift(1).expanding().mean())
        + df.groupby("grid_id")["dropoff_count"].transform(lambda s: s.shift(1).expanding().mean())
    )
    df["net_flow_next_hour"] = df.groupby("grid_id")["net_flow"].shift(-1)
    df["pickup_count_next_hour"] = df.groupby("grid_id")["pickup_count"].shift(-1)
    df["dropoff_count_next_hour"] = df.groupby("grid_id")["dropoff_count"].shift(-1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df.to_csv(P.tables / "regional_spatial_features.csv", index=False)
    log(f"regional spatial feature dataset: {df.shape}")
    return df, {"subway_stations": subway_stations, "gbfs_stations": gbfs_stations, "pois": pois, "spatial": spatial}


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), 1)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    smape = np.mean(2 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1)) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "sMAPE": smape}


def train_models(df: pd.DataFrame, quick: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object], List[str]]:
    exclude = {
        "datetime_hour",
        "grid_id",
        "net_flow_next_hour",
        "pickup_count_next_hour",
        "dropoff_count_next_hour",
    }
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    ordered_times = np.array(sorted(df["datetime_hour"].unique()))
    split_idx = int(len(ordered_times) * (1 - CONFIG["test_size"]))
    split_time = ordered_times[split_idx]
    train = df[df["datetime_hour"] < split_time].copy()
    test = df[df["datetime_hour"] >= split_time].copy()
    fit_train = train
    max_train_rows = 220_000 if quick else int(CONFIG["max_train_rows"])
    if len(train) > max_train_rows:
        sample_idx = np.linspace(0, len(train) - 1, max_train_rows, dtype=int)
        fit_train = train.iloc[sample_idx].copy()
        log(f"training rows capped from {len(train):,} to {len(fit_train):,}; test metrics still use full time holdout")
    X_train = fit_train[feature_cols].to_numpy()
    y_train = fit_train["net_flow_next_hour"].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_test = test["net_flow_next_hour"].to_numpy()

    models = {
        "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Random Forest": RandomForestRegressor(
            n_estimators=60 if quick else 120,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=CONFIG["random_state"],
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=80 if quick else 180,
            learning_rate=0.06,
            l2_regularization=0.01,
            random_state=CONFIG["random_state"],
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(96, 48),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=80 if quick else 140,
                        early_stopping=True,
                        random_state=CONFIG["random_state"],
                    ),
                ),
            ]
        ),
    }

    fitted = {}
    metric_rows = []
    pred_df = test[["datetime_hour", "grid_id", "grid_lat", "grid_lng", "net_flow_next_hour", "historical_avg_demand"]].copy()
    pred_df = pred_df.rename(columns={"net_flow_next_hour": "actual_net_flow_next_hour"})
    for name, model in models.items():
        log(f"fit {name}")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred_df[name] = pred
        metric_rows.append({"model": name, **regression_metrics(y_test, pred)})
        fitted[name] = model
        with (P.models / f"regional_{name.replace(' ', '_').lower()}.pkl").open("wb") as f:
            pickle.dump(model, f)

    metrics_df = pd.DataFrame(metric_rows).sort_values("RMSE")
    metrics_df.to_csv(P.tables / "regional_model_metrics.csv", index=False)
    pred_df.to_csv(P.tables / "regional_predictions.csv", index=False)
    log(f"model metrics saved: {P.tables / 'regional_model_metrics.csv'}")
    return metrics_df, pred_df, fitted, feature_cols


def build_dispatch_risk(pred_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    threshold = float(train_df["net_flow_next_hour"].abs().quantile(0.75))
    latest_time = pred_df["datetime_hour"].max()
    latest = pred_df[pred_df["datetime_hour"] == latest_time].copy()
    latest["predicted_net_flow_next_hour"] = latest["HistGradientBoosting"]
    latest["risk_type"] = np.select(
        [
            latest["predicted_net_flow_next_hour"] < -threshold,
            latest["predicted_net_flow_next_hour"] > threshold,
        ],
        ["shortage_risk", "overflow_risk"],
        default="normal",
    )
    latest["dispatch_priority"] = latest["predicted_net_flow_next_hour"].abs() * latest["historical_avg_demand"].fillna(0)
    shortage = latest[latest["risk_type"] == "shortage_risk"].sort_values("dispatch_priority", ascending=False).head(10)
    overflow = latest[latest["risk_type"] == "overflow_risk"].sort_values("dispatch_priority", ascending=False).head(10)
    risk = pd.concat([shortage, overflow], ignore_index=True)
    cols = [
        "datetime_hour",
        "grid_id",
        "grid_lat",
        "grid_lng",
        "risk_type",
        "predicted_net_flow_next_hour",
        "actual_net_flow_next_hour",
        "historical_avg_demand",
        "dispatch_priority",
    ]
    risk[cols].to_csv(P.tables / "dispatch_risk_top10.csv", index=False)
    log(f"dispatch risk threshold={threshold:.3f}, rows={len(risk)}")
    return risk[cols]


def make_visualizations(
    df: pd.DataFrame,
    aux: Dict[str, pd.DataFrame],
    metrics_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    risk_df: pd.DataFrame,
) -> None:
    log("make regional figures")
    grid_summary = df.groupby(["grid_id", "grid_lat", "grid_lng"], as_index=False).agg(
        pickup_count=("pickup_count", "sum"),
        dropoff_count=("dropoff_count", "sum"),
        avg_abs_net_flow=("net_flow", lambda s: float(np.mean(np.abs(s)))),
        station_count=("station_count", "max"),
    )
    grid_summary["demand"] = grid_summary["pickup_count"] + grid_summary["dropoff_count"]

    plt.figure(figsize=(8, 8))
    sc = plt.scatter(grid_summary["grid_lng"], grid_summary["grid_lat"], c=grid_summary["demand"], s=9, cmap="viridis", alpha=0.85)
    plt.colorbar(sc, label="Total pickups + dropoffs")
    plt.title("区域网格地图：Citi Bike 活跃区域需求")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    savefig("regional_grid_map.png")

    plt.figure(figsize=(8, 8))
    base = plt.scatter(grid_summary["grid_lng"], grid_summary["grid_lat"], c=grid_summary["station_count"], s=10, cmap="Blues", alpha=0.75)
    plt.colorbar(base, label="GBFS station count by grid")
    subway = aux.get("subway_stations", pd.DataFrame())
    if not subway.empty:
        plt.scatter(subway["lng"], subway["lat"], s=12, c="#E15759", marker="^", label="MTA subway station", alpha=0.85)
        plt.legend(loc="best")
    plt.title("地铁站与 Citi Bike 区域叠加图")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    savefig("subway_bike_grid_map.png")

    poi_cols = [c for c in df.columns if c.endswith("_count_500m")]
    poi_density = df.groupby(["grid_id", "grid_lat", "grid_lng"], as_index=False)[poi_cols].max()
    poi_density["poi_count_500m_total"] = poi_density[poi_cols].sum(axis=1)
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(
        poi_density["grid_lng"],
        poi_density["grid_lat"],
        c=poi_density["poi_count_500m_total"],
        s=10,
        cmap="magma",
        alpha=0.85,
    )
    plt.colorbar(sc, label="POIs within 500m")
    plt.title("POI 密度地图")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    savefig("poi_density_map.png")

    latest_time = pred_df["datetime_hour"].max()
    latest = pred_df[pred_df["datetime_hour"] == latest_time].copy()
    latest["predicted_net_flow_next_hour"] = latest["HistGradientBoosting"]
    vmax = max(1.0, float(latest["predicted_net_flow_next_hour"].abs().quantile(0.98)))
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(
        latest["grid_lng"],
        latest["grid_lat"],
        c=latest["predicted_net_flow_next_hour"],
        s=15,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        alpha=0.9,
    )
    plt.colorbar(sc, label="Predicted net flow next hour")
    if not risk_df.empty:
        shortage = risk_df[risk_df["risk_type"] == "shortage_risk"]
        overflow = risk_df[risk_df["risk_type"] == "overflow_risk"]
        plt.scatter(shortage["grid_lng"], shortage["grid_lat"], s=70, facecolors="none", edgecolors="black", label="Top shortage")
        plt.scatter(overflow["grid_lng"], overflow["grid_lat"], s=70, facecolors="none", edgecolors="#F28E2B", label="Top overflow")
        plt.legend(loc="best")
    plt.title(f"预测净流量地图：{pd.Timestamp(latest_time)}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    savefig("predicted_net_flow_map.png")

    plt.figure(figsize=(9, 4.5))
    metric_long = metrics_df.melt(id_vars="model", value_vars=["MAE", "RMSE", "sMAPE"], var_name="metric", value_name="value")
    sns.barplot(data=metric_long, x="model", y="value", hue="metric")
    plt.title("模型指标对比图")
    plt.xticks(rotation=15)
    savefig("regional_model_metrics_bar.png")


def generate_report(metrics_df: pd.DataFrame, risk_df: pd.DataFrame, df: pd.DataFrame, aux: Dict[str, pd.DataFrame]) -> None:
    metric_md = metrics_df.to_markdown(index=False)
    shortage = risk_df[risk_df["risk_type"] == "shortage_risk"]
    overflow = risk_df[risk_df["risk_type"] == "overflow_risk"]
    shortage_md = shortage.to_markdown(index=False) if not shortage.empty else "暂无超过阈值的缺车风险区域。"
    overflow_md = overflow.to_markdown(index=False) if not overflow.empty else "暂无超过阈值的满桩风险区域。"
    summary_rows = [
        ["建模样本数", len(df)],
        ["区域数量", df["grid_id"].nunique()],
        ["时间范围", f"{df['datetime_hour'].min()} 至 {df['datetime_hour'].max()}"],
        ["MTA 地铁站数量", len(aux.get("subway_stations", pd.DataFrame()))],
        ["GBFS 站点数量", len(aux.get("gbfs_stations", pd.DataFrame()))],
        ["OSM/Overpass POI 数量", len(aux.get("pois", pd.DataFrame()))],
    ]
    summary_md = pd.DataFrame(summary_rows, columns=["指标", "数值"]).to_markdown(index=False)
    best = metrics_df.iloc[0]
    report = f"""# Citi Bike 区域级地图与空间特征调度风险分析

## 摘要

本项目将原先的城市级订单预测改造为区域级共享单车调度风险分析。城市级订单量只能告诉我们全市下一小时需求大概是多少，却无法回答“哪些区域会缺车、哪些区域会满桩”。调度问题的核心是空间不平衡，因此本次使用 Citi Bike 起终点经纬度划分网格区域，并预测每个区域下一小时净流量。

## 数据与样本

{summary_md}

## 为什么城市级预测不能直接指导调度

城市级订单预测会把曼哈顿、布鲁克林、皇后区等区域的流入流出抵消掉。例如 A 区大量取车、B 区大量还车时，全市订单量可能稳定，但 A 区会缺车、B 区会满桩。调度车辆需要知道空间位置、风险方向和优先级，而不是单一总量。

## 为什么引入空间特征

共享单车需求受周边功能强烈影响。地铁站附近通勤换乘明显，餐饮和商业 POI 影响午间与晚间活动，学校、公园和住宅区对应不同出行节奏，实时站点容量和空桩数则直接影响下一小时的服务风险。因此区域级模型同时使用历史净流量、时间特征、地图 POI、地铁站距离和 GBFS 实时站点状态。

## 地图与空间数据源

- Citi Bike tripdata：使用 `start_lat`、`start_lng`、`end_lat`、`end_lng` 构造区域级 pickup、dropoff 和 net flow。
- MTA Subway Stations 开放数据：计算最近地铁站距离、500 米和 1000 米地铁站数量。
- OpenStreetMap / Overpass API：统计每个区域 500 米范围内 shop、restaurant、cafe、school、park、residential、commercial 数量，并缓存 API 结果。
- Citi Bike GBFS：读取 station_information 和 station_status，统计每个区域站点数、总容量、当前可用车辆、空桩和电单车数量。

## 区域净流量定义

经纬度按 `{CONFIG['grid_size']}` 度固定网格聚合，每个 `grid_id` 表示一个区域：

`net_flow = dropoff_count - pickup_count`

主预测目标为：

`net_flow_next_hour = net_flow.groupby(grid_id).shift(-1)`

当预测净流量为负，说明下一小时该区域取车多于还车，可能缺车；当预测净流量为正，说明还车多于取车，可能满桩。

## 模型结果

按时间顺序切分训练集和测试集，未使用随机切分。对比 Linear Regression、Random Forest、HistGradientBoosting 和 MLP，主模型采用 HistGradientBoosting。本次最佳 RMSE 模型为 **{best['model']}**，RMSE={best['RMSE']:.3f}，R2={best['R2']:.4f}。

{metric_md}

![模型指标对比图](figures/regional_model_metrics_bar.png)

## 调度风险规则

阈值使用训练集中 `abs(net_flow_next_hour)` 的 75% 分位数。若 `predicted_net_flow_next_hour < -threshold`，标记为 `shortage_risk`；若 `predicted_net_flow_next_hour > threshold`，标记为 `overflow_risk`。

`dispatch_priority = abs(predicted_net_flow_next_hour) * historical_avg_demand`

该优先级同时考虑风险强度和历史需求规模。

## Top 10 缺车风险区域

{shortage_md}

## Top 10 满桩风险区域

{overflow_md}

## 可视化

![区域网格地图](figures/regional_grid_map.png)

![地铁站与 Citi Bike 区域叠加图](figures/subway_bike_grid_map.png)

![POI 密度地图](figures/poi_density_map.png)

![预测净流量地图](figures/predicted_net_flow_map.png)

## 局限性

1. GBFS 是实时状态，只代表脚本运行时的站点状态，不等同于 2026 年历史每小时真实库存。
2. POI 数据质量依赖 OpenStreetMap，存在分类不一致和覆盖不完整的问题。
3. 当前模型没有纳入真实调度车辆路径、车辆载重、人工成本和作业时窗。
4. 固定网格比真实运营区简单，边界附近会出现空间归属误差。
5. Overpass 或 GBFS 请求失败时会使用离线降级特征，保证流程运行，但空间解释力会下降。

## 交付文件

- `outputs/tables/regional_spatial_features.csv`
- `outputs/tables/regional_model_metrics.csv`
- `outputs/tables/dispatch_risk_top10.csv`
- `outputs/figures/regional_grid_map.png`
- `outputs/figures/subway_bike_grid_map.png`
- `outputs/figures/poi_density_map.png`
- `outputs/figures/predicted_net_flow_map.png`
- `outputs/report_regional_spatial.md`
"""
    (ROOT / "outputs" / "report_regional_spatial.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer chunks and model iterations for smoke tests.")
    parser.add_argument("--skip-download", action="store_true", help="Use local Citi Bike zip files.")
    args = parser.parse_args()
    P.ensure()
    quick = bool(args.quick or CONFIG["quick"])
    months = CONFIG["months"]
    zips = [P.raw_citibike / f"{m}-citibike-tripdata.zip" for m in months] if args.skip_download else download_citibike(months)
    missing = [str(p) for p in zips if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Citi Bike zip files: {missing}")
    flow = build_regional_flow(zips, quick=quick)
    df, aux = build_model_dataset(flow, quick=quick)
    metrics_df, pred_df, fitted, feature_cols = train_models(df, quick=quick)
    ordered_times = np.array(sorted(df["datetime_hour"].unique()))
    split_idx = int(len(ordered_times) * (1 - CONFIG["test_size"]))
    train_df = df[df["datetime_hour"] < ordered_times[split_idx]].copy()
    risk_df = build_dispatch_risk(pred_df, train_df)
    make_visualizations(df, aux, metrics_df, pred_df, risk_df)
    generate_report(metrics_df, risk_df, df, aux)
    log("done")
    log(f"features: {P.tables / 'regional_spatial_features.csv'}")
    log(f"metrics: {P.tables / 'regional_model_metrics.csv'}")
    log(f"risk: {P.tables / 'dispatch_risk_top10.csv'}")
    log(f"report: {ROOT / 'outputs' / 'report_regional_spatial.md'}")


if __name__ == "__main__":
    main()
