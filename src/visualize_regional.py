from __future__ import annotations

import os
from typing import Dict

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .regional_common import PATHS, ensure_dirs, log


sns.set_theme(style="whitegrid", font="Arial Unicode MS")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(PATHS["figures"] / name, bbox_inches="tight")
    plt.close()


def scatter_map(df: pd.DataFrame, value: str, title: str, filename: str, cmap: str = "viridis", center: bool = False) -> None:
    plt.figure(figsize=(7.5, 7))
    kwargs = {}
    if center:
        vmax = max(1.0, float(df[value].abs().quantile(0.98)))
        kwargs.update({"vmin": -vmax, "vmax": vmax})
    sc = plt.scatter(
        df["grid_center_lng"],
        df["grid_center_lat"],
        c=df[value],
        s=70,
        cmap=cmap,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.25,
        **kwargs,
    )
    plt.colorbar(sc, label=value)
    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    savefig(filename)


def label_top_points(df: pd.DataFrame, value: str, count: int = 5) -> None:
    if df.empty or value not in df.columns:
        return
    for row in df.sort_values(value, ascending=False).head(count).itertuples(index=False):
        plt.annotate(
            str(row.grid_id),
            (row.grid_center_lng, row.grid_center_lat),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )


def make_subway_overlay_map(region_info: pd.DataFrame) -> None:
    map_path = PATHS["tables"] / "region_map_features.csv"
    station_path = PATHS["tables"] / "subway_station_points.csv"
    if not map_path.exists():
        return
    map_features = pd.read_csv(map_path)
    stations = pd.read_csv(station_path) if station_path.exists() else pd.DataFrame()
    merged = region_info.merge(
        map_features[["grid_id", "nearest_subway_distance", "subway_count_500m", "transit_congestion_index"]],
        on="grid_id",
        how="left",
    )
    lng_min, lng_max = merged["grid_center_lng"].min() - 0.035, merged["grid_center_lng"].max() + 0.035
    lat_min, lat_max = merged["grid_center_lat"].min() - 0.035, merged["grid_center_lat"].max() + 0.035
    if not stations.empty and {"lat", "lng"}.issubset(stations.columns):
        stations = stations[
            stations["lng"].between(lng_min, lng_max)
            & stations["lat"].between(lat_min, lat_max)
        ]
    plt.figure(figsize=(8, 7.2))
    sc = plt.scatter(
        merged["grid_center_lng"],
        merged["grid_center_lat"],
        c=merged["subway_count_500m"].fillna(0),
        s=np.clip(merged["total_events"] / max(merged["total_events"].max(), 1) * 180, 35, 180),
        cmap="YlGnBu",
        edgecolors="#263238",
        linewidths=0.25,
        alpha=0.88,
        label="Citi Bike grid",
    )
    if not stations.empty and {"lat", "lng"}.issubset(stations.columns):
        plt.scatter(
            stations["lng"],
            stations["lat"],
            s=18,
            c="#C62828",
            marker="^",
            alpha=0.72,
            label="MTA subway station",
        )
    label_top_points(merged, "transit_congestion_index", count=3)
    plt.colorbar(sc, label="Subway stations within 500m")
    plt.title("Subway stations and Citi Bike regional grids")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="best")
    plt.xlim(lng_min, lng_max)
    plt.ylim(lat_min, lat_max)
    plt.axis("equal")
    savefig("subway_bike_grid_map.png")


def make_congestion_risk_map(region_info: pd.DataFrame, pred: pd.DataFrame) -> None:
    map_path = PATHS["tables"] / "region_map_features.csv"
    latest_time = pred["datetime"].max()
    latest = pred[pred["datetime"] == latest_time].rename(columns={"y_pred_net_flow": "predicted_net_flow"})
    latest = latest.merge(region_info[["grid_id", "grid_center_lat", "grid_center_lng", "total_events"]], on="grid_id", how="left")
    if map_path.exists():
        map_features = pd.read_csv(map_path)
        latest = latest.merge(map_features[["grid_id", "transit_congestion_index", "subway_count_500m"]], on="grid_id", how="left")
    if "transit_congestion_index" not in latest.columns:
        latest["transit_congestion_index"] = 0
    if "subway_count_500m" not in latest.columns:
        latest["subway_count_500m"] = 0
    latest["transit_congestion_index"] = latest["transit_congestion_index"].fillna(0)
    latest["subway_count_500m"] = latest["subway_count_500m"].fillna(0)
    latest["congestion_score"] = (
        latest["predicted_net_flow"].clip(lower=0)
        * (latest["total_events"].fillna(0) / max(latest["total_events"].max(), 1))
        * (1.0 + latest["transit_congestion_index"] / 10.0)
    )
    plt.figure(figsize=(8, 7.2))
    plt.scatter(region_info["grid_center_lng"], region_info["grid_center_lat"], s=32, c="#ECEFF1", edgecolors="#B0BEC5", linewidths=0.2)
    sc = plt.scatter(
        latest["grid_center_lng"],
        latest["grid_center_lat"],
        c=latest["congestion_score"],
        s=np.clip(np.abs(latest["predicted_net_flow"]) * 10 + 45, 45, 260),
        cmap="Reds",
        edgecolors="#263238",
        linewidths=0.35,
        alpha=0.92,
    )
    label_top_points(latest, "congestion_score", count=6)
    plt.colorbar(sc, label="Congestion score")
    plt.title(f"Predicted overflow congestion risk at {latest_time}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis("equal")
    savefig("congestion_risk_map.png")


def make_visualizations(config: Dict) -> None:
    ensure_dirs()
    panel = pd.read_csv(PATHS["processed"] / "regional_hourly_panel.csv")
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    region_info = pd.read_csv(PATHS["tables"] / "region_grid_info.csv")
    metrics = pd.read_csv(PATHS["tables"] / "regional_model_metrics.csv")
    pred = pd.read_csv(PATHS["tables"] / "stgcn_predictions.csv")
    pred["datetime"] = pd.to_datetime(pred["datetime"])

    scatter_map(region_info, "total_events", "Regional grid distribution", "regional_grid_map.png", "viridis")
    make_subway_overlay_map(region_info)

    high_demand = panel.copy()
    high_demand["total_demand"] = high_demand["pickup_count"] + high_demand["dropoff_count"]
    peak_time = high_demand.groupby("datetime")["total_demand"].sum().idxmax()
    peak = high_demand[high_demand["datetime"] == peak_time]
    scatter_map(peak, "pickup_count", f"Pickup heatmap at {peak_time}", "regional_pickup_heatmap.png", "YlOrRd")
    scatter_map(peak, "dropoff_count", f"Dropoff heatmap at {peak_time}", "regional_dropoff_heatmap.png", "YlGnBu")
    scatter_map(peak, "net_flow", f"Actual net flow at {peak_time}", "regional_net_flow_map.png", "RdBu", center=True)

    latest_time = pred["datetime"].max()
    latest_pred = pred[pred["datetime"] == latest_time].merge(
        region_info[["grid_id", "grid_center_lat", "grid_center_lng"]], on="grid_id", how="left"
    )
    latest_pred = latest_pred.rename(columns={"y_pred_net_flow": "predicted_net_flow"})
    scatter_map(
        latest_pred,
        "predicted_net_flow",
        f"STGCN predicted next-hour net flow at {latest_time}",
        "stgcn_predicted_net_flow_map.png",
        "RdBu",
        center=True,
    )

    risk_path = PATHS["tables"] / "dispatch_risk_top10.csv"
    risk = pd.read_csv(risk_path) if risk_path.exists() else pd.DataFrame()
    plt.figure(figsize=(7.5, 7))
    plt.scatter(region_info["grid_center_lng"], region_info["grid_center_lat"], s=40, c="#D7DCE2", label="balanced/background")
    if not risk.empty:
        colors = {"shortage_risk": "#E15759", "overflow_risk": "#4C78A8"}
        for risk_type, group in risk.groupby("risk_type"):
            plt.scatter(
                group["grid_center_lng"],
                group["grid_center_lat"],
                s=120,
                c=colors.get(risk_type, "#999999"),
                edgecolors="black",
                label=risk_type,
            )
    plt.title("Dispatch risk map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc="best")
    plt.axis("equal")
    savefig("dispatch_risk_map.png")
    make_congestion_risk_map(region_info, pred)

    metric_long = metrics.melt(id_vars=["model"], value_vars=["MAE", "RMSE", "R2"], var_name="metric", value_name="value")
    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=metric_long, x="model", y="value", hue="metric")
    plt.title("Regional model metrics comparison")
    plt.xticks(rotation=15)
    savefig("regional_model_metrics_bar.png")

    top_nodes = (
        panel.groupby("grid_id")
        .agg(total_demand=("pickup_count", "sum"))
        .sort_values("total_demand", ascending=False)
        .head(5)
        .index.tolist()
    )
    plot_pred = pred[pred["grid_id"].isin(top_nodes)].copy()
    if len(plot_pred) > 5 * 120:
        keep_times = sorted(plot_pred["datetime"].unique())[-120:]
        plot_pred = plot_pred[plot_pred["datetime"].isin(keep_times)]
    fig, axes = plt.subplots(len(top_nodes), 1, figsize=(11, 2.2 * len(top_nodes)), sharex=True)
    if len(top_nodes) == 1:
        axes = [axes]
    for ax, grid_id in zip(axes, top_nodes):
        g = plot_pred[plot_pred["grid_id"] == grid_id]
        ax.plot(g["datetime"], g["y_true_net_flow"], label="actual", color="black", lw=1.2)
        ax.plot(g["datetime"], g["y_pred_net_flow"], label="STGCN", color="#4C78A8", lw=1.2)
        ax.set_title(grid_id, loc="left", fontsize=10)
        ax.axhline(0, color="#999999", lw=0.8)
    axes[0].legend(loc="best")
    fig.suptitle("STGCN actual vs predicted net flow: Top 5 demand regions", y=1.01)
    savefig("stgcn_actual_vs_predicted.png")

    loss = pd.read_csv(PATHS["tables"] / "stgcn_training_loss.csv")
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(loss["epoch"], loss["train_loss"], marker="o", label="train")
    plt.plot(loss["epoch"], loss["val_loss"], marker="o", label="validation")
    plt.title("STGCN training loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    savefig("stgcn_training_loss.png")
    log("saved regional STGCN figures")
