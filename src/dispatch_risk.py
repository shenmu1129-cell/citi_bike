from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .build_map_features import MAP_FEATURE_COLUMNS
from .regional_common import PATHS, ensure_dirs, log


def compute_threshold(config: Dict, train_panel: pd.DataFrame) -> float:
    configured = config.get("threshold", "auto")
    if configured != "auto":
        return float(configured)
    return float(train_panel["net_flow_next_hour"].abs().quantile(0.75))


def build_dispatch_risk(config: Dict) -> pd.DataFrame:
    ensure_dirs()
    pred = pd.read_csv(PATHS["tables"] / "stgcn_predictions.csv")
    pred["datetime"] = pd.to_datetime(pred["datetime"])
    model_df = pd.read_csv(PATHS["processed"] / "regional_model_dataset.csv")
    model_df["datetime"] = pd.to_datetime(model_df["datetime"])
    times = np.array(sorted(model_df["datetime"].unique()))
    train_end = int(len(times) * float(config["train_ratio"]))
    train_panel = model_df[model_df["datetime"] < times[train_end]]
    threshold = compute_threshold(config, train_panel)
    region_info = pd.read_csv(PATHS["tables"] / "region_grid_info.csv")
    demand = model_df.groupby("grid_id", as_index=False).agg(
        historical_avg_demand=("pickup_count", "mean"),
        historical_avg_dropoff=("dropoff_count", "mean"),
    )
    demand["historical_avg_demand"] = demand["historical_avg_demand"] + demand["historical_avg_dropoff"]
    scored = pred.copy().rename(columns={"y_pred_net_flow": "predicted_net_flow"})
    scored["risk_type"] = np.select(
        [
            scored["predicted_net_flow"] < -threshold,
            scored["predicted_net_flow"] > threshold,
        ],
        ["shortage_risk", "overflow_risk"],
        default="balanced",
    )
    scored = scored.merge(region_info[["grid_id", "grid_center_lat", "grid_center_lng"]], on="grid_id", how="left")
    scored = scored.merge(demand[["grid_id", "historical_avg_demand"]], on="grid_id", how="left")
    map_path = PATHS["tables"] / "region_map_features.csv"
    if map_path.exists():
        map_features = pd.read_csv(map_path)
        keep_cols = ["grid_id"] + [c for c in MAP_FEATURE_COLUMNS if c in map_features.columns]
        scored = scored.merge(map_features[keep_cols], on="grid_id", how="left")
    for col in MAP_FEATURE_COLUMNS:
        if col not in scored.columns:
            scored[col] = 0.0
    scored["nearest_subway_distance"] = scored["nearest_subway_distance"].fillna(9999.0)
    for col in [c for c in MAP_FEATURE_COLUMNS if c != "nearest_subway_distance"]:
        scored[col] = scored[col].fillna(0.0)
    scored["dispatch_priority"] = scored["predicted_net_flow"].abs() * scored["historical_avg_demand"].fillna(0)
    scored["congestion_score"] = (
        scored["predicted_net_flow"].clip(lower=0)
        * scored["historical_avg_demand"].fillna(0)
        * (1.0 + scored["transit_congestion_index"].fillna(0) / 10.0)
    )

    shortage = scored[scored["risk_type"] == "shortage_risk"].sort_values("dispatch_priority", ascending=False).head(10)
    overflow = scored[scored["risk_type"] == "overflow_risk"].sort_values("dispatch_priority", ascending=False).head(10)
    if len(shortage) < 10:
        shortage_fill = scored.sort_values("predicted_net_flow", ascending=True).head(10)
        shortage = pd.concat([shortage, shortage_fill], ignore_index=True).drop_duplicates(["datetime", "grid_id"]).head(10)
        shortage["risk_type"] = "shortage_risk"
    if len(overflow) < 10:
        overflow_fill = scored.sort_values("predicted_net_flow", ascending=False).head(10)
        overflow = pd.concat([overflow, overflow_fill], ignore_index=True).drop_duplicates(["datetime", "grid_id"]).head(10)
        overflow["risk_type"] = "overflow_risk"

    risk = pd.concat([shortage, overflow], ignore_index=True)
    risk["suggested_action"] = risk["risk_type"].map(
        {
            "shortage_risk": "建议提前补车",
            "overflow_risk": "建议提前移走车辆或预留空桩，缓解满桩拥堵",
            "balanced": "暂不需要明显调度",
        }
    )
    cols = [
        "datetime",
        "grid_id",
        "grid_center_lat",
        "grid_center_lng",
        "predicted_net_flow",
        "risk_type",
        "dispatch_priority",
        "congestion_score",
        "nearest_subway_distance",
        "subway_count_500m",
        "subway_count_1000m",
        "suggested_action",
    ]
    risk[cols].to_csv(PATHS["tables"] / "dispatch_risk_top10.csv", index=False)
    congestion = (
        scored[scored["predicted_net_flow"] > 0]
        .sort_values(["congestion_score", "predicted_net_flow"], ascending=False)
        .head(10)
        .copy()
    )
    congestion["risk_type"] = "overflow_congestion_risk"
    congestion["suggested_action"] = "建议提前移走车辆或预留空桩，优先保障地铁/商业周边区域"
    congestion[cols].to_csv(PATHS["tables"] / "congestion_risk_top10.csv", index=False)
    log(f"saved dispatch risk top10, threshold={threshold:.3f}, rows={len(risk)}")
    return risk[cols]
