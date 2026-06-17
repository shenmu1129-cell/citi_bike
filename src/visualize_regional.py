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


def make_visualizations(config: Dict) -> None:
    ensure_dirs()
    panel = pd.read_csv(PATHS["processed"] / "regional_hourly_panel.csv")
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    region_info = pd.read_csv(PATHS["tables"] / "region_grid_info.csv")
    metrics = pd.read_csv(PATHS["tables"] / "regional_model_metrics.csv")
    pred = pd.read_csv(PATHS["tables"] / "stgcn_predictions.csv")
    pred["datetime"] = pd.to_datetime(pred["datetime"])

    scatter_map(region_info, "total_events", "Regional grid distribution", "regional_grid_map.png", "viridis")

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
