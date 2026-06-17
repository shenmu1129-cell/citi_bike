from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests

from .build_map_features import MAP_FEATURE_COLUMNS
from .regional_common import PATHS, ensure_dirs, log, zip_paths_for_months


USECOLS = {
    "ride_id",
    "rideable_type",
    "started_at",
    "ended_at",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "member_casual",
}


def download_months(months: List[str], config: Dict, skip_download: bool = False) -> List[Path]:
    ensure_dirs()
    paths = zip_paths_for_months(months)
    if skip_download:
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing Citi Bike zip files: {missing}")
        return paths
    for month, out in zip(months, paths):
        if out.exists() and out.stat().st_size > 10_000_000:
            log(f"skip existing {out.name}")
            continue
        url = str(config["citibike_url_template"]).format(month=month)
        tmp = out.with_suffix(out.suffix + ".part")
        for attempt in range(1, 5):
            existing = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}
            mode = "ab" if existing > 0 else "wb"
            try:
                action = "resume" if existing > 0 else "download"
                log(f"{action} {url} (attempt {attempt})")
                with requests.get(url, stream=True, timeout=120, headers=headers) as r:
                    if existing > 0 and r.status_code == 200:
                        mode = "wb"
                        existing = 0
                    r.raise_for_status()
                    with tmp.open(mode) as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                tmp.replace(out)
                break
            except Exception as exc:
                if attempt >= 4:
                    raise
                log(f"download interrupted for {month}: {exc}; retrying")
    return paths


def read_zip_chunks(zip_path: Path, quick: bool, quick_chunks_per_csv: int) -> Iterable[pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV in {zip_path}")
        for name in csv_names:
            log(f"read {zip_path.name}/{name}")
            with zf.open(name) as f:
                reader = pd.read_csv(
                    f,
                    usecols=lambda c: c in USECOLS,
                    chunksize=300_000,
                    low_memory=False,
                )
                for idx, chunk in enumerate(reader):
                    yield chunk
                    if quick and idx + 1 >= quick_chunks_per_csv:
                        log("quick mode: stop early for this CSV")
                        break


def grid_from_coords(lat: pd.Series, lng: pd.Series, grid_size: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    grid_lat = (pd.to_numeric(lat, errors="coerce") / grid_size).round() * grid_size
    grid_lng = (pd.to_numeric(lng, errors="coerce") / grid_size).round() * grid_size
    grid_id = grid_lat.map(lambda x: f"{x:.2f}") + "_" + grid_lng.map(lambda x: f"{x:.2f}")
    return grid_lat, grid_lng, grid_id


def valid_coords(df: pd.DataFrame) -> pd.Series:
    cols = ["start_lat", "start_lng", "end_lat", "end_lng"]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    return (
        numeric["start_lat"].between(40.45, 41.05)
        & numeric["end_lat"].between(40.45, 41.05)
        & numeric["start_lng"].between(-74.35, -73.55)
        & numeric["end_lng"].between(-74.35, -73.55)
    )


def clean_chunk(chunk: pd.DataFrame, grid_size: float) -> pd.DataFrame:
    chunk = chunk.dropna(subset=["ride_id", "started_at", "ended_at", "start_lat", "start_lng", "end_lat", "end_lng"])
    chunk = chunk.drop_duplicates("ride_id").copy()
    chunk["started_at"] = pd.to_datetime(chunk["started_at"], errors="coerce")
    chunk["ended_at"] = pd.to_datetime(chunk["ended_at"], errors="coerce")
    chunk = chunk.dropna(subset=["started_at", "ended_at"])
    chunk = chunk[valid_coords(chunk)].copy()
    duration = (chunk["ended_at"] - chunk["started_at"]).dt.total_seconds() / 60
    chunk = chunk[(duration >= 1) & (duration <= 1440)].copy()
    chunk["pickup_hour"] = chunk["started_at"].dt.floor("h")
    chunk["dropoff_hour"] = chunk["ended_at"].dt.floor("h")
    chunk["start_grid_lat"], chunk["start_grid_lng"], chunk["start_grid_id"] = grid_from_coords(
        chunk["start_lat"], chunk["start_lng"], grid_size
    )
    chunk["end_grid_lat"], chunk["end_grid_lng"], chunk["end_grid_id"] = grid_from_coords(
        chunk["end_lat"], chunk["end_lng"], grid_size
    )
    chunk["pickup_member_count"] = (chunk["member_casual"].astype(str) == "member").astype(int)
    chunk["pickup_casual_count"] = (chunk["member_casual"].astype(str) == "casual").astype(int)
    rideable = chunk["rideable_type"].astype(str)
    chunk["pickup_electric_count"] = rideable.str.contains("electric", case=False, na=False).astype(int)
    chunk["pickup_classic_count"] = rideable.str.contains("classic", case=False, na=False).astype(int)
    return chunk


def collect_regional_events(zip_paths: List[Path], config: Dict, quick: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grid_size = float(config["grid_size"])
    quick_chunks = int(config.get("quick_chunks_per_csv", 1))
    pickup_parts = []
    dropoff_parts = []
    demand_parts = []
    total_rows = 0
    for zip_path in zip_paths:
        for chunk in read_zip_chunks(zip_path, quick=quick, quick_chunks_per_csv=quick_chunks):
            total_rows += len(chunk)
            chunk = clean_chunk(chunk, grid_size)
            if chunk.empty:
                continue
            pickup_parts.append(
                chunk.groupby(["pickup_hour", "start_grid_id"], as_index=False).agg(
                    pickup_count=("ride_id", "count"),
                    pickup_member_count=("pickup_member_count", "sum"),
                    pickup_casual_count=("pickup_casual_count", "sum"),
                    pickup_electric_count=("pickup_electric_count", "sum"),
                    pickup_classic_count=("pickup_classic_count", "sum"),
                )
            )
            dropoff_parts.append(
                chunk.groupby(["dropoff_hour", "end_grid_id"], as_index=False).agg(dropoff_count=("ride_id", "count"))
            )
            demand_parts.append(chunk[["start_grid_id", "end_grid_id"]])
    if not pickup_parts or not dropoff_parts:
        raise RuntimeError("No valid regional records were built from Citi Bike input.")
    pickups = (
        pd.concat(pickup_parts, ignore_index=True)
        .rename(columns={"pickup_hour": "datetime", "start_grid_id": "grid_id"})
        .groupby(["datetime", "grid_id"], as_index=False)
        .sum()
    )
    dropoffs = (
        pd.concat(dropoff_parts, ignore_index=True)
        .rename(columns={"dropoff_hour": "datetime", "end_grid_id": "grid_id"})
        .groupby(["datetime", "grid_id"], as_index=False)
        .sum()
    )
    demand = pd.concat(demand_parts, ignore_index=True)
    grid_counts = pd.concat(
        [
            demand["start_grid_id"].rename("grid_id"),
            demand["end_grid_id"].rename("grid_id"),
        ],
        ignore_index=True,
    ).value_counts()
    log(f"regional events from raw rows: {total_rows:,}")
    return pickups, dropoffs, grid_counts.rename("total_events").reset_index()


def parse_grid_info(grid_ids: pd.Series, grid_counts: pd.DataFrame) -> pd.DataFrame:
    parts = grid_ids.astype(str).str.split("_", expand=True)
    info = pd.DataFrame(
        {
            "grid_id": grid_ids.to_numpy(),
            "grid_center_lat": parts[0].astype(float).to_numpy(),
            "grid_center_lng": parts[1].astype(float).to_numpy(),
        }
    )
    return info.merge(grid_counts, on="grid_id", how="left").fillna({"total_events": 0})


def build_regional_panel(zip_paths: List[Path], config: Dict, quick: bool = False) -> pd.DataFrame:
    ensure_dirs()
    pickups, dropoffs, grid_counts = collect_regional_events(zip_paths, config, quick=quick)
    top_n = int(config.get("top_regions", 40))
    if len(grid_counts) < top_n:
        top_n = max(int(config.get("min_regions", 30)), len(grid_counts))
    selected = grid_counts.sort_values("total_events", ascending=False).head(top_n)["grid_id"]
    selected_set = set(selected)
    od_pairs = []
    # Re-scan only compact start/end grid IDs to build an OD graph among selected regions.
    # This keeps the graph extension independent from the hourly panel aggregation.
    grid_size = float(config["grid_size"])
    quick_chunks = int(config.get("quick_chunks_per_csv", 1))
    for zip_path in zip_paths:
        for chunk in read_zip_chunks(zip_path, quick=quick, quick_chunks_per_csv=quick_chunks):
            chunk = clean_chunk(chunk, grid_size)
            if chunk.empty:
                continue
            pairs = chunk[["start_grid_id", "end_grid_id"]].copy()
            pairs = pairs[pairs["start_grid_id"].isin(selected_set) & pairs["end_grid_id"].isin(selected_set)]
            if not pairs.empty:
                od_pairs.append(pairs)
    if od_pairs:
        od = (
            pd.concat(od_pairs, ignore_index=True)
            .groupby(["start_grid_id", "end_grid_id"], as_index=False)
            .size()
            .rename(columns={"start_grid_id": "source_grid_id", "end_grid_id": "target_grid_id", "size": "trip_count"})
        )
    else:
        od = pd.DataFrame(columns=["source_grid_id", "target_grid_id", "trip_count"])
    od.to_csv(PATHS["tables"] / "region_od_edges.csv", index=False)
    hours = pd.date_range(
        min(pickups["datetime"].min(), dropoffs["datetime"].min()),
        max(pickups["datetime"].max(), dropoffs["datetime"].max()),
        freq="h",
    )
    panel = pd.MultiIndex.from_product([hours, selected], names=["datetime", "grid_id"]).to_frame(index=False)
    panel = panel.merge(pickups, on=["datetime", "grid_id"], how="left").merge(dropoffs, on=["datetime", "grid_id"], how="left")
    count_cols = [
        "pickup_count",
        "pickup_member_count",
        "pickup_casual_count",
        "pickup_electric_count",
        "pickup_classic_count",
        "dropoff_count",
    ]
    panel[count_cols] = panel[count_cols].fillna(0).astype("int32")
    panel["net_flow"] = panel["dropoff_count"] - panel["pickup_count"]
    region_info = parse_grid_info(selected.reset_index(drop=True), grid_counts)
    panel = panel.merge(region_info[["grid_id", "grid_center_lat", "grid_center_lng"]], on="grid_id", how="left")
    panel = panel.sort_values(["datetime", "grid_id"]).reset_index(drop=True)
    panel.to_csv(PATHS["processed"] / "regional_hourly_panel.csv", index=False)
    region_info.to_csv(PATHS["tables"] / "region_grid_info.csv", index=False)
    log(f"saved regional panel: {panel.shape}, regions={panel['grid_id'].nunique()}")
    return panel


def build_regional_model_dataset(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_dirs()
    if panel is None:
        panel = pd.read_csv(PATHS["processed"] / "regional_hourly_panel.csv")
    panel = panel.copy()
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    df = panel.sort_values(["grid_id", "datetime"]).reset_index(drop=True)
    map_path = PATHS["tables"] / "region_map_features.csv"
    if map_path.exists():
        map_features = pd.read_csv(map_path)
        keep_cols = ["grid_id"] + [c for c in MAP_FEATURE_COLUMNS if c in map_features.columns]
        df = df.merge(map_features[keep_cols], on="grid_id", how="left")
    for col in MAP_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    df["nearest_subway_distance"] = df["nearest_subway_distance"].fillna(9999.0)
    for col in [c for c in MAP_FEATURE_COLUMNS if c != "nearest_subway_distance"]:
        df[col] = df[col].fillna(0.0)
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    for lag in [1, 2, 3, 24, 168]:
        df[f"pickup_lag_{lag}"] = df.groupby("grid_id")["pickup_count"].shift(lag)
    for lag in [1, 24, 168]:
        df[f"dropoff_lag_{lag}"] = df.groupby("grid_id")["dropoff_count"].shift(lag)
        df[f"net_flow_lag_{lag}"] = df.groupby("grid_id")["net_flow"].shift(lag)

    df["pickup_rolling_mean_3"] = df.groupby("grid_id")["pickup_count"].transform(lambda s: s.shift(1).rolling(3).mean())
    df["pickup_rolling_mean_24"] = df.groupby("grid_id")["pickup_count"].transform(lambda s: s.shift(1).rolling(24).mean())
    df["pickup_rolling_mean_168"] = df.groupby("grid_id")["pickup_count"].transform(lambda s: s.shift(1).rolling(168).mean())
    df["dropoff_rolling_mean_24"] = df.groupby("grid_id")["dropoff_count"].transform(lambda s: s.shift(1).rolling(24).mean())
    df["net_flow_rolling_mean_24"] = df.groupby("grid_id")["net_flow"].transform(lambda s: s.shift(1).rolling(24).mean())
    df["net_flow_rolling_std_24"] = df.groupby("grid_id")["net_flow"].transform(lambda s: s.shift(1).rolling(24).std())

    df["pickup_count_next_hour"] = df.groupby("grid_id")["pickup_count"].shift(-1)
    df["dropoff_count_next_hour"] = df.groupby("grid_id")["dropoff_count"].shift(-1)
    df["net_flow_next_hour"] = df.groupby("grid_id")["net_flow"].shift(-1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df.to_csv(PATHS["processed"] / "regional_model_dataset.csv", index=False)
    log(f"saved regional model dataset: {df.shape}")
    return df
