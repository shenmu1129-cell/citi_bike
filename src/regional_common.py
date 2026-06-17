from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIG: Dict[str, Any] = {
    "months": [],
    "citibike_url_template": "https://s3.amazonaws.com/tripdata/{month}-citibike-tripdata.zip",
    "grid_size": 0.01,
    "top_regions": 40,
    "min_regions": 30,
    "knn_k": 5,
    "od_k": 5,
    "sigma_m": 1500.0,
    "distance_graph_weight": 0.75,
    "od_graph_weight": 0.25,
    "lookback": 24,
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_state": 42,
    "quick_months": 2,
    "quick_chunks_per_csv": 1,
    "quick_epochs": 25,
    "epochs": 80,
    "patience": 8,
    "batch_size": 32,
    "learning_rate": 0.001,
    "hidden_channels": 64,
    "threshold": "auto",
}


PATHS = {
    "raw_citibike": ROOT / "data" / "raw" / "citibike",
    "processed": ROOT / "data" / "processed",
    "figures": ROOT / "outputs" / "figures",
    "tables": ROOT / "outputs" / "tables",
    "models": ROOT / "outputs" / "models",
    "cache": ROOT / ".cache",
}


def ensure_dirs() -> None:
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
    (ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_value(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return ""
    if raw in {"true", "True"}:
        return True
    if raw in {"false", "False"}:
        return False
    if raw in {"null", "None"}:
        return None
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [parse_value(x.strip().strip("'\"")) for x in inner.split(",")]
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    path = config_path or (ROOT / "config_stgcn.yaml")
    if not path.exists():
        return config
    current_key = None
    list_values: List[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_key:
            list_values.append(parse_value(stripped[2:]))
            config[current_key] = list_values
            continue
        current_key = None
        list_values = []
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            current_key = key
            config[key] = []
            list_values = []
        else:
            config[key] = parse_value(value)
    return config


def month_range_ending(latest_month: str, count: int) -> List[str]:
    ts = pd.Timestamp(f"{latest_month[:4]}-{latest_month[4:]}-01")
    months = [(ts - pd.DateOffset(months=i)).strftime("%Y%m") for i in range(count)]
    return sorted(months)


def local_citibike_months() -> List[str]:
    months = []
    for path in PATHS["raw_citibike"].glob("*-citibike-tripdata.zip"):
        token = path.name.split("-citibike-tripdata.zip")[0]
        if len(token) == 6 and token.isdigit():
            months.append(token)
    return sorted(set(months))


def resolve_months(config: Dict[str, Any], quick: bool = False) -> List[str]:
    configured = [str(m) for m in config.get("months", []) if str(m).strip()]
    local = local_citibike_months()
    if configured:
        months = sorted(configured)
    elif local:
        months = local[-12:]
    else:
        latest = (pd.Timestamp.today().replace(day=1) - pd.DateOffset(months=1)).strftime("%Y%m")
        months = month_range_ending(latest, 12)
    if quick:
        keep = int(config.get("quick_months", 2))
        months = months[-keep:]
    return months


def zip_paths_for_months(months: Iterable[str]) -> List[Path]:
    return [PATHS["raw_citibike"] / f"{month}-citibike-tripdata.zip" for month in months]


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


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), 1.0)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    smape = float(np.mean(2 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1.0)) * 100)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "MAPE": mape, "sMAPE": smape}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
