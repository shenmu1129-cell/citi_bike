#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font="Arial Unicode MS")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close()


def load_features() -> pd.DataFrame:
    cols = [
        "datetime_hour",
        "grid_id",
        "pickup_count",
        "dropoff_count",
        "net_flow",
        "hour",
        "weekday",
        "nearest_subway_distance",
        "subway_count_500m",
        "shop_count_500m",
        "restaurant_count_500m",
        "cafe_count_500m",
        "school_count_500m",
        "park_count_500m",
        "residential_count_500m",
        "commercial_count_500m",
        "station_count",
        "total_capacity",
        "current_bikes",
        "current_empty_docks",
    ]
    df = pd.read_csv(TABLE_DIR / "regional_spatial_features.csv", usecols=cols)
    df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
    return df


def make_data_figures(df: pd.DataFrame) -> None:
    hourly = df.groupby("datetime_hour", as_index=False).agg(
        pickup_count=("pickup_count", "sum"),
        dropoff_count=("dropoff_count", "sum"),
        abs_net_flow=("net_flow", lambda s: float(np.abs(s).sum())),
    )
    plt.figure(figsize=(11, 4.5))
    plt.plot(hourly["datetime_hour"], hourly["pickup_count"], label="Pickups", lw=1.2)
    plt.plot(hourly["datetime_hour"], hourly["dropoff_count"], label="Dropoffs", lw=1.2)
    plt.title("区域聚合后的每小时取车与还车趋势")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.legend()
    savefig("regional_hourly_pickup_dropoff_trend.png")

    clipped = df["net_flow"].clip(df["net_flow"].quantile(0.01), df["net_flow"].quantile(0.99))
    plt.figure(figsize=(8, 4.5))
    sns.histplot(clipped, bins=60, kde=True, color="#4C78A8")
    plt.axvline(0, color="black", lw=1, ls="--")
    plt.title("区域小时净流量分布")
    plt.xlabel("Net flow = dropoffs - pickups")
    plt.ylabel("Region-hour samples")
    savefig("regional_net_flow_distribution.png")

    heat = df.pivot_table(index="weekday", columns="hour", values="net_flow", aggfunc=lambda s: float(np.abs(s).mean()))
    plt.figure(figsize=(10.5, 4.5))
    sns.heatmap(heat, cmap="YlOrRd")
    plt.title("星期-小时平均绝对净流量热力图")
    plt.xlabel("Hour")
    plt.ylabel("Weekday, 0=Monday")
    savefig("regional_weekday_hour_abs_net_flow_heatmap.png")


def make_experiment_figures() -> None:
    metrics = pd.read_csv(TABLE_DIR / "regional_model_metrics.csv")
    predictions = pd.read_csv(TABLE_DIR / "regional_predictions.csv")
    predictions["datetime_hour"] = pd.to_datetime(predictions["datetime_hour"])
    model_cols = ["Linear Regression", "Random Forest", "HistGradientBoosting", "MLP"]

    order = metrics.sort_values("RMSE")["model"].tolist()
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    sns.barplot(data=metrics, x="model", y="RMSE", order=order, color="#4C78A8", ax=ax1)
    ax1.set_ylabel("RMSE")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=15)
    ax2 = ax1.twinx()
    sns.lineplot(data=metrics.set_index("model").loc[order].reset_index(), x="model", y="R2", marker="o", color="#E15759", ax=ax2)
    ax2.set_ylabel("R2")
    ax2.grid(False)
    plt.title("模型实验对比：RMSE 与 R2")
    savefig("regional_model_rmse_r2_comparison.png")

    sample = predictions.sample(min(80_000, len(predictions)), random_state=42)
    plt.figure(figsize=(5.8, 5.4))
    sns.scatterplot(
        data=sample,
        x="actual_net_flow_next_hour",
        y="HistGradientBoosting",
        s=10,
        alpha=0.28,
        edgecolor=None,
    )
    q_low = float(sample[["actual_net_flow_next_hour", "HistGradientBoosting"]].quantile(0.01).min())
    q_high = float(sample[["actual_net_flow_next_hour", "HistGradientBoosting"]].quantile(0.99).max())
    plt.plot([q_low, q_high], [q_low, q_high], color="red", ls="--", lw=1)
    plt.xlim(q_low, q_high)
    plt.ylim(q_low, q_high)
    plt.title("HistGradientBoosting 真实值 vs 预测值")
    plt.xlabel("Actual net flow next hour")
    plt.ylabel("Predicted net flow next hour")
    savefig("regional_hgb_actual_vs_predicted_scatter.png")

    err_rows = []
    for col in model_cols:
        err = (predictions[col] - predictions["actual_net_flow_next_hour"]).abs()
        err = err.clip(upper=err.quantile(0.98))
        err_rows.append(pd.DataFrame({"model": col, "absolute_error": err}))
    err_df = pd.concat(err_rows, ignore_index=True)
    if len(err_df) > 160_000:
        err_df = err_df.sample(160_000, random_state=42)
    plt.figure(figsize=(9, 4.5))
    sns.boxplot(data=err_df, x="model", y="absolute_error", showfliers=False, color="#A0CBE8")
    plt.title("模型绝对误差分布对比")
    plt.xlabel("")
    plt.ylabel("Absolute error")
    plt.xticks(rotation=15)
    savefig("regional_model_absolute_error_boxplot.png")


def make_risk_and_feature_figures(df: pd.DataFrame) -> None:
    risk = pd.read_csv(TABLE_DIR / "dispatch_risk_top10.csv")
    risk["label"] = risk["risk_type"].map({"shortage_risk": "Shortage", "overflow_risk": "Overflow"}) + " " + risk["grid_id"]
    risk = risk.sort_values("dispatch_priority", ascending=True)
    colors = risk["risk_type"].map({"shortage_risk": "#E15759", "overflow_risk": "#4C78A8"}).tolist()
    plt.figure(figsize=(9, 6.5))
    plt.barh(risk["label"], risk["dispatch_priority"], color=colors)
    plt.title("Top 调度风险区域优先级对比")
    plt.xlabel("Dispatch priority")
    plt.ylabel("")
    savefig("regional_top_dispatch_priority_bar.png")

    spatial_cols = [
        "nearest_subway_distance",
        "subway_count_500m",
        "shop_count_500m",
        "restaurant_count_500m",
        "cafe_count_500m",
        "school_count_500m",
        "park_count_500m",
        "residential_count_500m",
        "commercial_count_500m",
        "station_count",
        "total_capacity",
        "current_bikes",
        "current_empty_docks",
    ]
    grid = df.groupby("grid_id", as_index=False).agg(
        avg_abs_net_flow=("net_flow", lambda s: float(np.abs(s).mean())),
        **{col: (col, "max") for col in spatial_cols},
    )
    corr = grid[spatial_cols + ["avg_abs_net_flow"]].corr(numeric_only=True)["avg_abs_net_flow"].drop("avg_abs_net_flow")
    corr = corr.reindex(corr.abs().sort_values(ascending=True).index)
    plt.figure(figsize=(8.5, 5.5))
    corr.plot(kind="barh", color=np.where(corr >= 0, "#59A14F", "#E15759"))
    plt.axvline(0, color="black", lw=1)
    plt.title("空间特征与区域平均绝对净流量的相关性")
    plt.xlabel("Pearson correlation")
    savefig("regional_spatial_feature_correlation_bar.png")


def main() -> None:
    df = load_features()
    make_data_figures(df)
    make_experiment_figures()
    make_risk_and_feature_figures(df)
    print("presentation figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
