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
from typing import Dict, Iterable, List, Tuple

import matplotlib

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


CONFIG = {
    "months": ["202603", "202604", "202605"],
    "citibike_url_template": "https://s3.amazonaws.com/tripdata/{month}-citibike-tripdata.zip",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "timezone": "America/New_York",
    "test_size": 0.2,
    "random_state": 42,
    "quick": False,
}


@dataclass
class Paths:
    root: Path = ROOT
    raw_citibike: Path = ROOT / "data" / "raw" / "citibike"
    raw_weather: Path = ROOT / "data" / "raw" / "weather"
    interim: Path = ROOT / "data" / "interim"
    processed: Path = ROOT / "data" / "processed"
    figures: Path = ROOT / "outputs" / "figures"
    tables: Path = ROOT / "outputs" / "tables"
    models: Path = ROOT / "outputs" / "models"
    cache: Path = ROOT / ".cache"

    def ensure(self) -> None:
        for p in [
            self.raw_citibike,
            self.raw_weather,
            self.interim,
            self.processed,
            self.figures,
            self.tables,
            self.models,
            self.cache,
            self.cache / "matplotlib",
        ]:
            p.mkdir(parents=True, exist_ok=True)


P = Paths()
sns.set_theme(style="whitegrid", font="Arial Unicode MS")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 140


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_config() -> None:
    (ROOT / "config.json").write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")


def download_file(url: str, out: Path, timeout: int = 60) -> None:
    if out.exists() and out.stat().st_size > 10_000_000:
        log(f"skip existing {out.name} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
        return
    log(f"download {url}")
    tmp = out.with_suffix(out.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total and done // (100 * 1024 * 1024) != (done - len(chunk)) // (100 * 1024 * 1024):
                    log(f"  {out.name}: {done / total:.1%} ({done / 1024 / 1024:.0f} MB)")
    tmp.replace(out)
    log(f"saved {out.name} ({out.stat().st_size / 1024 / 1024:.1f} MB)")


def download_citibike(months: List[str]) -> List[Path]:
    P.ensure()
    zips = []
    for month in months:
        url = CONFIG["citibike_url_template"].format(month=month)
        out = P.raw_citibike / f"{month}-citibike-tripdata.zip"
        download_file(url, out)
        zips.append(out)
    return zips


def read_zip_csvs(zip_path: Path, quick: bool = False) -> Iterable[pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No CSV in {zip_path}")
        for name in names:
            log(f"read {zip_path.name}/{name}")
            with zf.open(name) as f:
                reader = pd.read_csv(
                    f,
                    usecols=lambda c: c in {
                        "ride_id",
                        "rideable_type",
                        "started_at",
                        "ended_at",
                        "member_casual",
                    },
                    chunksize=300_000,
                    low_memory=False,
                )
                for i, chunk in enumerate(reader):
                    yield chunk
                    if quick and i >= 1:
                        log("quick mode: stop after two chunks per csv")
                        break


def preprocess_citibike(zips: List[Path], quick: bool = False) -> pd.DataFrame:
    records = []
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
            chunk["datetime"] = chunk["started_at"].dt.floor("h")
            chunk["casual_count"] = (chunk["member_casual"] == "casual").astype(int)
            chunk["member_count"] = (chunk["member_casual"] == "member").astype(int)
            chunk["electric_count"] = chunk["rideable_type"].astype(str).str.contains("electric", case=False, na=False).astype(int)
            chunk["classic_count"] = chunk["rideable_type"].astype(str).str.contains("classic", case=False, na=False).astype(int)
            g = chunk.groupby("datetime", as_index=False).agg(
                rental_count=("ride_id", "count"),
                casual_count=("casual_count", "sum"),
                member_count=("member_count", "sum"),
                electric_count=("electric_count", "sum"),
                classic_count=("classic_count", "sum"),
            )
            records.append(g)
    hourly = pd.concat(records, ignore_index=True)
    hourly = hourly.groupby("datetime", as_index=False).sum().sort_values("datetime")
    hourly.to_csv(P.interim / "hourly_rides.csv", index=False)
    log(f"hourly rides: {hourly.shape}, raw rows read: {total_rows}")
    return hourly


def download_weather(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    start_date = start.strftime("%Y-%m-%d")
    end_date = end.strftime("%Y-%m-%d")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": CONFIG["latitude"],
        "longitude": CONFIG["longitude"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "rain",
                "wind_speed_10m",
                "cloud_cover",
            ]
        ),
        "timezone": CONFIG["timezone"],
    }
    raw_path = P.raw_weather / "weather_raw.json"
    if raw_path.exists():
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if raw.get("hourly"):
            log("skip existing weather_raw.json")
            return raw
    log(f"download weather {start_date} to {end_date}")
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    raw = r.json()
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return raw


def preprocess_weather(raw: dict) -> pd.DataFrame:
    hourly = raw["hourly"]
    df = pd.DataFrame(hourly).rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).drop_duplicates("datetime").sort_values("datetime")
    for col in df.columns:
        if col != "datetime":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.set_index("datetime").interpolate(method="time").ffill().bfill().reset_index()
    df["is_rain"] = (df["precipitation"] > 0).astype(int)
    df["is_heavy_rain"] = (df["precipitation"] >= 2).astype(int)
    df.to_csv(P.interim / "hourly_weather.csv", index=False)
    log(f"hourly weather: {df.shape}")
    return df


def build_features(rides: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    rides = rides.copy()
    weather = weather.copy()
    rides["datetime"] = pd.to_datetime(rides["datetime"])
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    start = max(rides["datetime"].min(), weather["datetime"].min())
    end = min(rides["datetime"].max(), weather["datetime"].max())
    full = pd.DataFrame({"datetime": pd.date_range(start, end, freq="h")})
    df = full.merge(rides, on="datetime", how="left").merge(weather, on="datetime", how="left")
    count_cols = ["rental_count", "casual_count", "member_count", "electric_count", "classic_count"]
    df[count_cols] = df[count_cols].fillna(0)
    weather_cols = [c for c in weather.columns if c != "datetime"]
    df[weather_cols] = df[weather_cols].interpolate().ffill().bfill()
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    for lag in [1, 2, 3, 24, 168]:
        df[f"lag_{lag}"] = df["rental_count"].shift(lag)
    for win in [3, 6, 24]:
        df[f"rolling_mean_{win}"] = df["rental_count"].shift(1).rolling(win).mean()
    df["rolling_std_24"] = df["rental_count"].shift(1).rolling(24).std()
    df["target_next_hour"] = df["rental_count"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    df.to_csv(P.processed / "model_dataset.csv", index=False)
    log(f"model dataset: {df.shape}, {df.datetime.min()} to {df.datetime.max()}")
    return df


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(P.figures / name, bbox_inches="tight")
    plt.close()


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temperature_bin"] = pd.cut(out["temperature_2m"], bins=7)
    out["precipitation_level"] = pd.cut(
        out["precipitation"],
        bins=[-0.001, 0, 0.5, 2, np.inf],
        labels=["无降水", "小雨", "中雨", "强降水"],
    )
    out["wind_bin"] = pd.cut(out["wind_speed_10m"], bins=5)
    out["humidity_bin"] = pd.cut(out["relative_humidity_2m"], bins=5)
    out["scene"] = np.select(
        [
            (out["is_rush_hour"] == 1) & (out["is_rain"] == 1),
            (out["is_rush_hour"] == 1) & (out["is_rain"] == 0),
            (out["is_rush_hour"] == 0) & (out["is_rain"] == 1),
        ],
        ["高峰+降雨", "高峰+无雨", "非高峰+降雨"],
        default="非高峰+无雨",
    )
    return out


def make_eda(df: pd.DataFrame) -> None:
    log("make EDA figures")
    edf = add_bins(df)
    summary = pd.DataFrame(
        [
            ["样本时间范围", f"{df.datetime.min()} 到 {df.datetime.max()}"],
            ["总小时数", len(df)],
            ["总骑行量", int(df["rental_count"].sum())],
            ["平均小时骑行量", round(float(df["rental_count"].mean()), 2)],
            ["最大小时骑行量", int(df["rental_count"].max())],
            ["缺失值数量", int(df.isna().sum().sum())],
            ["下雨小时占比", round(float(df["is_rain"].mean()), 4)],
            ["周末小时占比", round(float(df["is_weekend"].mean()), 4)],
        ],
        columns=["指标", "数值"],
    )
    summary.to_csv(P.tables / "eda_summary.csv", index=False)

    plt.figure(figsize=(13, 4))
    plt.plot(df["datetime"], df["rental_count"], lw=0.8)
    plt.title("每小时 Citi Bike 骑行需求时间序列")
    plt.xlabel("时间")
    plt.ylabel("骑行量")
    savefig("01_hourly_demand_timeseries.png")

    plt.figure(figsize=(8, 4))
    df.groupby("hour")["rental_count"].mean().plot(kind="bar", color="#4C78A8")
    plt.title("不同小时平均骑行量")
    plt.xlabel("小时")
    plt.ylabel("平均骑行量")
    savefig("02_avg_demand_by_hour.png")

    plt.figure(figsize=(5, 4))
    df.groupby("is_weekend")["rental_count"].mean().rename(index={0: "工作日", 1: "周末"}).plot(kind="bar", color=["#59A14F", "#F28E2B"])
    plt.title("工作日与周末平均骑行量")
    savefig("03_weekday_vs_weekend.png")

    plt.figure(figsize=(7, 4))
    df.groupby("weekday")["rental_count"].mean().plot(kind="bar", color="#59A14F")
    plt.title("不同星期平均骑行量")
    plt.xlabel("星期：0=周一")
    savefig("04_avg_demand_by_weekday.png")

    for fig_name, xcol, title in [
        ("05_temperature_scatter.png", "temperature_2m", "温度与骑行需求散点图"),
        ("07_wind_scatter.png", "wind_speed_10m", "风速与骑行需求散点图"),
    ]:
        plt.figure(figsize=(7, 4))
        sns.scatterplot(data=df, x=xcol, y="target_next_hour", s=14, alpha=0.45)
        plt.title(title)
        savefig(fig_name)

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=edf, x="precipitation_level", y="rental_count", color="#A0CBE8")
    plt.title("不同降水强度下骑行量分布")
    savefig("06_precipitation_boxplot.png")

    corr_cols = [
        "target_next_hour",
        "rental_count",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "hour",
        "weekday",
        "is_weekend",
        "is_rush_hour",
        "lag_1",
        "lag_24",
        "rolling_mean_24",
    ]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
    plt.title("主要变量相关性热力图")
    savefig("08_correlation_heatmap.png")

    plt.figure(figsize=(7, 4))
    sns.histplot(df["target_next_hour"], bins=40, kde=True, color="#4C78A8")
    plt.title("目标变量分布")
    savefig("09_target_distribution.png")

    n_train = int(len(df) * (1 - CONFIG["test_size"]))
    plt.figure(figsize=(12, 4))
    plt.plot(df["datetime"].iloc[:n_train], df["rental_count"].iloc[:n_train], label="训练集")
    plt.plot(df["datetime"].iloc[n_train:], df["rental_count"].iloc[n_train:], label="测试集")
    plt.title("训练集与测试集时间顺序切分")
    plt.legend()
    savefig("10_train_test_split.png")

    heat = df.pivot_table(index="weekday", columns="hour", values="rental_count", aggfunc="mean")
    plt.figure(figsize=(11, 4.5))
    sns.heatmap(heat, cmap="YlOrRd")
    plt.title("星期-小时平均骑行需求热力图")
    savefig("18_weekday_hour_demand_heatmap.png")

    date_heat = df.copy()
    date_heat["date"] = date_heat["datetime"].dt.date
    heat2 = date_heat.pivot_table(index="date", columns="hour", values="rental_count", aggfunc="sum")
    plt.figure(figsize=(11, 8))
    sns.heatmap(heat2, cmap="mako")
    plt.title("日期-小时骑行需求热力图")
    savefig("19_date_hour_demand_heatmap.png")

    daily = df.set_index("datetime").resample("D").agg(
        rental_count=("rental_count", "sum"),
        temperature_2m=("temperature_2m", "mean"),
        precipitation=("precipitation", "sum"),
    )
    plt.figure(figsize=(12, 4))
    plt.plot(daily.index, daily["rental_count"], label="日骑行量")
    plt.plot(daily.index, daily["rental_count"].rolling(7).mean(), label="7日滚动均值", lw=2)
    plt.title("每日骑行量及 7 日滚动趋势")
    plt.legend()
    savefig("20_daily_demand_rolling_trend.png")

    for fig_name, cols, title in [
        ("21_member_casual_stacked_by_hour.png", ["member_count", "casual_count"], "不同小时会员与临时用户平均骑行量"),
        ("22_rideable_type_stacked_by_hour.png", ["classic_count", "electric_count"], "不同小时车型平均使用量"),
    ]:
        hour_mean = df.groupby("hour")[cols].mean()
        plt.figure(figsize=(9, 4))
        hour_mean.plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title(title)
        savefig(fig_name)

    for fig_name, col, title in [
        ("23_temperature_bin_bar.png", "temperature_bin", "不同温度区间的平均骑行量"),
        ("24_precipitation_intensity_bar.png", "precipitation_level", "不同降水强度下的平均骑行量"),
        ("25_wind_speed_bin_bar.png", "wind_bin", "不同风速区间的平均骑行量"),
        ("26_humidity_bin_bar.png", "humidity_bin", "不同湿度区间的平均骑行量"),
        ("30_rush_hour_rain_scene_bar.png", "scene", "高峰期与降雨组合场景下的平均骑行量"),
    ]:
        plt.figure(figsize=(8, 4))
        edf.groupby(col, observed=False)["rental_count"].mean().plot(kind="bar", color="#E15759")
        plt.title(title)
        plt.xticks(rotation=25, ha="right")
        savefig(fig_name)

    for fig_name, lag, title in [
        ("27_lag1_target_scatter.png", "lag_1", "上一小时需求与下一小时需求关系"),
        ("28_lag24_target_scatter.png", "lag_24", "前一天同小时需求与下一小时需求关系"),
    ]:
        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=df, x=lag, y="target_next_hour", s=14, alpha=0.45)
        plt.title(title)
        savefig(fig_name)

    acfs = [df["rental_count"].autocorr(lag=i) for i in range(1, 49)]
    plt.figure(figsize=(9, 4))
    plt.plot(range(1, 49), acfs, marker="o", ms=3)
    plt.axvline(24, color="red", ls="--", label="24小时")
    plt.title("骑行需求自相关曲线")
    plt.xlabel("滞后小时")
    plt.ylabel("自相关")
    plt.legend()
    savefig("29_demand_autocorrelation_curve.png")

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(daily.index, daily["rental_count"], color="#4C78A8", label="日骑行量")
    ax2 = ax1.twinx()
    ax2.plot(daily.index, daily["temperature_2m"], color="#F28E2B", label="平均气温")
    ax2.bar(daily.index, daily["precipitation"], color="#59A14F", alpha=0.25, label="日降水")
    ax1.set_title("日骑行量、气温与降水对照曲线")
    savefig("31_daily_demand_weather_overlay.png")


class TorchMLP:
    def __init__(self, epochs: int = 500, patience: int = 40, lr: float = 1e-3, random_state: int = 42):
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.loss_history_: List[Tuple[int, float, float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        if torch is None:
            raise RuntimeError("PyTorch is not available")
        torch.manual_seed(self.random_state)
        Xs = self.scaler.fit_transform(X).astype("float32")
        ys = y.reshape(-1, 1).astype("float32")
        split = int(len(Xs) * 0.85)
        train_ds = TensorDataset(torch.tensor(Xs[:split]), torch.tensor(ys[:split]))
        val_x = torch.tensor(Xs[split:])
        val_y = torch.tensor(ys[split:])
        loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        self.model = nn.Sequential(
            nn.Linear(Xs.shape[1], 96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 1),
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        best = math.inf
        best_state = None
        wait = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses = []
            for bx, by in loader:
                pred = self.model(bx)
                loss = loss_fn(pred, by)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_losses.append(float(loss.item()))
            self.model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model(val_x), val_y).item())
            self.loss_history_.append((epoch, float(np.mean(train_losses)), val_loss))
            if val_loss < best:
                best = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= self.patience:
                break
        if best_state:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X).astype("float32")
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(Xs)).numpy().reshape(-1)


def metrics(y_true, y_pred) -> Dict[str, float]:
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), 1)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    smape = np.mean(2 * np.abs(y_pred - y_true) / np.maximum(np.abs(y_true) + np.abs(y_pred), 1)) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "sMAPE": smape}


def train_models(df: pd.DataFrame, quick: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, str, List[str]]:
    log("train models")
    feature_cols = [
        c
        for c in df.columns
        if c not in {"datetime", "target_next_hour"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    n_train = int(len(df) * (1 - CONFIG["test_size"]))
    train, test = df.iloc[:n_train], df.iloc[n_train:]
    X_train = train[feature_cols].to_numpy()
    y_train = train["target_next_hour"].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_test = test["target_next_hour"].to_numpy()

    models = {
        "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Random Forest": RandomForestRegressor(
            n_estimators=80 if quick else 220,
            min_samples_leaf=2,
            random_state=CONFIG["random_state"],
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=80 if quick else 260,
            learning_rate=0.06,
            random_state=CONFIG["random_state"],
        ),
    }
    if torch is not None:
        models["MLP"] = TorchMLP(epochs=180 if quick else 600, patience=30 if quick else 60, random_state=CONFIG["random_state"])

    metric_rows = []
    pred_df = pd.DataFrame({"datetime": test["datetime"].to_numpy(), "actual": y_test})
    fitted = {}
    for name, model in models.items():
        log(f"fit {name}")
        model.fit(X_train, y_train)
        pred = np.maximum(model.predict(X_test), 0)
        pred_df[name] = pred
        metric_rows.append({"model": name, **metrics(y_test, pred)})
        fitted[name] = model
        with (P.models / f"{name.replace(' ', '_').lower()}.pkl").open("wb") as f:
            pickle.dump(model, f)
    metric_df = pd.DataFrame(metric_rows).sort_values("RMSE")
    metric_df.to_csv(P.tables / "model_metrics.csv", index=False)
    pred_df.to_csv(P.tables / "predictions.csv", index=False)
    best_model = metric_df.iloc[0]["model"]

    if "MLP" in fitted and getattr(fitted["MLP"], "loss_history_", None):
        hist = pd.DataFrame(fitted["MLP"].loss_history_, columns=["epoch", "train_loss", "val_loss"])
        hist.to_csv(P.tables / "mlp_loss_history.csv", index=False)
        plt.figure(figsize=(7, 4))
        plt.plot(hist["epoch"], hist["train_loss"], label="训练损失")
        plt.plot(hist["epoch"], hist["val_loss"], label="验证损失")
        plt.title("MLP 深度学习模型训练损失曲线")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        savefig("11_mlp_loss_curve.png")

    make_model_figures(df, pred_df, metric_df, best_model, fitted, feature_cols)
    return metric_df, pred_df, best_model, feature_cols


def make_model_figures(df, pred_df, metric_df, best_model, fitted, feature_cols) -> None:
    plt.figure(figsize=(10, 5))
    tmp = metric_df.melt(id_vars="model", value_vars=["MAE", "RMSE", "sMAPE"], var_name="指标", value_name="数值")
    sns.barplot(data=tmp, x="model", y="数值", hue="指标")
    plt.title("模型指标对比")
    plt.xticks(rotation=15)
    savefig("12_model_metrics_bar.png")

    plt.figure(figsize=(12, 4))
    plt.plot(pred_df["datetime"], pred_df["actual"], label="真实值", color="black", lw=1.6)
    for col in [c for c in pred_df.columns if c not in {"datetime", "actual"}]:
        plt.plot(pred_df["datetime"], pred_df[col], label=col, alpha=0.75)
    plt.title("所有模型真实值与预测值对比")
    plt.legend(ncol=3, fontsize=8)
    savefig("13_actual_vs_predicted_all_models.png")

    err_long = []
    for col in [c for c in pred_df.columns if c not in {"datetime", "actual"}]:
        err_long.extend([{"model": col, "error": p - a} for p, a in zip(pred_df[col], pred_df["actual"])])
    err_long = pd.DataFrame(err_long)
    plt.figure(figsize=(9, 4))
    sns.kdeplot(data=err_long, x="error", hue="model", common_norm=False)
    plt.title("预测误差分布")
    savefig("14_error_distribution.png")

    if "Random Forest" in fitted:
        imp = pd.Series(fitted["Random Forest"].feature_importances_, index=feature_cols).sort_values(ascending=False).head(20)
        plt.figure(figsize=(8, 6))
        imp.sort_values().plot(kind="barh", color="#4C78A8")
        plt.title("Random Forest 特征重要性 Top 20")
        savefig("15_random_forest_feature_importance.png")

    if "Linear Regression" in fitted:
        coefs = fitted["Linear Regression"].named_steps["model"].coef_
        coef = pd.Series(coefs, index=feature_cols).reindex(feature_cols).sort_values(key=np.abs, ascending=False).head(20)
        plt.figure(figsize=(8, 6))
        coef.sort_values().plot(kind="barh", color="#E15759")
        plt.title("Linear Regression 标准化系数 Top 20")
        savefig("16_linear_regression_coefficients.png")

    pred_df["best_pred"] = pred_df[best_model]
    pred_df["residual"] = pred_df["best_pred"] - pred_df["actual"]
    pred_df["abs_error"] = pred_df["residual"].abs()

    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=pred_df, x="actual", y="residual", s=18, alpha=0.6)
    plt.axhline(0, color="red", ls="--")
    plt.title("最佳模型残差图")
    savefig("17_best_model_residuals.png")

    plt.figure(figsize=(12, 4))
    plt.plot(pred_df["datetime"], pred_df["actual"], label="真实值", color="black", lw=1.5)
    plt.plot(pred_df["datetime"], pred_df["best_pred"], label=f"{best_model} 预测值", color="#4C78A8")
    plt.title("最佳模型真实值与预测值对比")
    plt.legend()
    savefig("32_best_model_actual_vs_predicted.png")

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=pred_df, x="actual", y="best_pred", s=20, alpha=0.65)
    mn = min(pred_df["actual"].min(), pred_df["best_pred"].min())
    mx = max(pred_df["actual"].max(), pred_df["best_pred"].max())
    plt.plot([mn, mx], [mn, mx], color="red", ls="--")
    plt.title("最佳模型预测值 vs 真实值散点图")
    savefig("33_best_model_predicted_vs_actual_scatter.png")

    pred_df["hour"] = pd.to_datetime(pred_df["datetime"]).dt.hour
    pred_df["weekday"] = pd.to_datetime(pred_df["datetime"]).dt.weekday
    plt.figure(figsize=(9, 4))
    sns.boxplot(data=pred_df, x="hour", y="abs_error", color="#A0CBE8")
    plt.title("最佳模型不同小时绝对误差分布")
    savefig("34_abs_error_by_hour_boxplot.png")

    test_context = df.iloc[int(len(df) * (1 - CONFIG["test_size"])):].copy().reset_index(drop=True)
    scene = np.select(
        [
            (test_context["is_rush_hour"] == 1) & (test_context["is_rain"] == 1),
            (test_context["is_rush_hour"] == 1) & (test_context["is_rain"] == 0),
            (test_context["is_rush_hour"] == 0) & (test_context["is_rain"] == 1),
        ],
        ["高峰+降雨", "高峰+无雨", "非高峰+降雨"],
        default="非高峰+无雨",
    )
    pred_df["scene"] = scene
    plt.figure(figsize=(7, 4))
    pred_df.groupby("scene")["abs_error"].mean().plot(kind="bar", color="#F28E2B")
    plt.title("不同场景下最佳模型平均绝对误差")
    plt.xticks(rotation=15)
    savefig("35_abs_error_by_rush_rain_scene.png")

    heat = pred_df.pivot_table(index="weekday", columns="hour", values="abs_error", aggfunc="mean")
    plt.figure(figsize=(10, 4))
    sns.heatmap(heat, cmap="Reds")
    plt.title("星期-小时预测绝对误差热力图")
    savefig("36_weekday_hour_error_heatmap.png")

    hour_metrics = []
    for col in [c for c in pred_df.columns if c not in {"datetime", "actual", "best_pred", "residual", "abs_error", "hour", "weekday", "scene"}]:
        for hour, g in pred_df.groupby("hour"):
            hour_metrics.append({"model": col, "hour": hour, "RMSE": mean_squared_error(g["actual"], g[col]) ** 0.5})
    hm = pd.DataFrame(hour_metrics)
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=hm, x="hour", y="RMSE", hue="model", marker="o")
    plt.title("不同模型按小时 RMSE 对比")
    savefig("37_hourly_rmse_by_model.png")


def generate_report(df: pd.DataFrame, metric_df: pd.DataFrame, best_model: str) -> None:
    log("generate markdown report")
    summary = pd.read_csv(P.tables / "eda_summary.csv")
    metric_md = metric_df.to_markdown(index=False)
    summary_md = summary.to_markdown(index=False)
    best = metric_df.iloc[0]
    report = f"""# 基于 Citi Bike 历史骑行数据与天气 API 的城市共享单车小时需求预测研究

## 摘要

本项目围绕城市共享单车小时需求预测问题，重新构建了完整的数据科学流程。项目使用纽约 Citi Bike 官方历史骑行数据，并结合 Open-Meteo Historical Weather API 的小时级天气数据，完成数据采集、清洗、小时聚合、天气融合、探索性分析、特征工程、模型训练、模型评估与报告输出。

研究任务定义为监督学习回归问题：给定当前小时的时间特征、天气特征和历史需求特征，预测下一小时全市 Citi Bike 骑行订单量。实验比较 Linear Regression、Random Forest、HistGradientBoosting 和 MLP 模型。本次重跑结果中，按 RMSE 排名的最佳模型为 **{best_model}**，RMSE={best['RMSE']:.3f}，R2={best['R2']:.6f}。

## 1. 数据来源

- Citi Bike 官方 tripdata：`202603`、`202604`、`202605`
- Open-Meteo Historical Weather API：纽约经纬度 40.7128, -74.0060
- 数据粒度：城市级小时需求
- 建模目标：`target_next_hour = rental_count.shift(-1)`

## 2. EDA 摘要

{summary_md}

## 3. 数据处理流程

1. 下载 Citi Bike 月度 zip 文件。
2. 分块读取 zip 内 CSV，清洗缺失值和异常骑行时长。
3. 按小时聚合订单量、会员/临时用户数量、车型数量。
4. 根据骑行数据时间范围请求 Open-Meteo 小时天气。
5. 按 `datetime` 合并骑行需求与天气变量。
6. 构造时间、周期、天气、历史滞后和滚动统计特征。
7. 按时间顺序切分训练集和测试集，避免数据泄漏。

## 4. 关键探索性分析图

![每小时 Citi Bike 骑行需求时间序列](figures/01_hourly_demand_timeseries.png)

![不同小时平均骑行量](figures/02_avg_demand_by_hour.png)

![星期-小时平均骑行需求热力图](figures/18_weekday_hour_demand_heatmap.png)

![日期-小时骑行需求热力图](figures/19_date_hour_demand_heatmap.png)

![每日骑行量及 7 日滚动趋势](figures/20_daily_demand_rolling_trend.png)

![不同小时会员与临时用户平均骑行量](figures/21_member_casual_stacked_by_hour.png)

![不同小时车型平均使用量](figures/22_rideable_type_stacked_by_hour.png)

## 5. 天气影响分析

![不同温度区间的平均骑行量](figures/23_temperature_bin_bar.png)

![不同降水强度下的平均骑行量](figures/24_precipitation_intensity_bar.png)

![高峰期与降雨组合场景下的平均骑行量](figures/30_rush_hour_rain_scene_bar.png)

![日骑行量、气温与降水对照曲线](figures/31_daily_demand_weather_overlay.png)

## 6. 特征工程与相关性

本项目构造五类特征：时间特征、周期编码、天气特征、历史滞后特征和滚动统计特征。历史需求特征如 `lag_1`、`lag_24`、`rolling_mean_24` 对预测下一小时需求具有重要意义。

![主要变量相关性热力图](figures/08_correlation_heatmap.png)

![上一小时需求与下一小时需求关系](figures/27_lag1_target_scatter.png)

![前一天同小时需求与下一小时需求关系](figures/28_lag24_target_scatter.png)

![骑行需求自相关曲线](figures/29_demand_autocorrelation_curve.png)

## 7. 模型结果

{metric_md}

![模型指标对比](figures/12_model_metrics_bar.png)

![所有模型真实值与预测值对比](figures/13_actual_vs_predicted_all_models.png)

![最佳模型真实值与预测值对比](figures/32_best_model_actual_vs_predicted.png)

![最佳模型预测值 vs 真实值散点图](figures/33_best_model_predicted_vs_actual_scatter.png)

## 8. 模型解释与误差诊断

![Random Forest 特征重要性 Top 20](figures/15_random_forest_feature_importance.png)

![Linear Regression 标准化系数 Top 20](figures/16_linear_regression_coefficients.png)

![最佳模型残差图](figures/17_best_model_residuals.png)

![预测误差分布](figures/14_error_distribution.png)

![最佳模型不同小时绝对误差分布](figures/34_abs_error_by_hour_boxplot.png)

![不同场景下最佳模型平均绝对误差](figures/35_abs_error_by_rush_rain_scene.png)

![星期-小时预测绝对误差热力图](figures/36_weekday_hour_error_heatmap.png)

![不同模型按小时 RMSE 对比](figures/37_hourly_rmse_by_model.png)

## 9. 结论

1. Citi Bike 小时需求具有明显的日内节律和周内差异。
2. 历史需求滞后特征和滚动统计特征对下一小时预测非常关键。
3. 降水、温度、风速、湿度等天气变量能够解释部分非周期波动。
4. 非线性模型整体优于线性基线，说明需求变化存在复杂交互关系。
5. 高峰时段和降雨场景仍是误差较大的薄弱区域。

## 10. 后续改进

- 从城市级预测扩展到站点级或区域级预测。
- 加入节假日、大型活动、地铁停运等外部事件变量。
- 尝试 LSTM、Transformer 或图神经网络。
- 将预测结果接入车辆调度优化模型。

## 交付物

- `run_all.py`：一键运行脚本
- `data/raw/`：原始 Citi Bike 与天气数据
- `data/processed/model_dataset.csv`：最终建模数据集
- `outputs/figures/`：分析和评估图表
- `outputs/tables/model_metrics.csv`：模型评估表
- `outputs/report.md`：本报告
"""
    (ROOT / "outputs" / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer chunks/iterations for smoke testing.")
    parser.add_argument("--skip-download", action="store_true", help="Use already downloaded raw files.")
    args = parser.parse_args()
    P.ensure()
    save_config()
    quick = bool(args.quick or CONFIG.get("quick"))
    months = CONFIG["months"]
    if args.skip_download:
        zips = [P.raw_citibike / f"{m}-citibike-tripdata.zip" for m in months]
    else:
        zips = download_citibike(months)
    rides = preprocess_citibike(zips, quick=quick)
    raw_weather = download_weather(rides["datetime"].min(), rides["datetime"].max())
    weather = preprocess_weather(raw_weather)
    df = build_features(rides, weather)
    make_eda(df)
    metric_df, pred_df, best_model, feature_cols = train_models(df, quick=quick)
    generate_report(df, metric_df, best_model)
    log("done")
    log(f"report: {ROOT / 'outputs' / 'report.md'}")
    log(f"metrics: {P.tables / 'model_metrics.csv'}")
    log(f"figures: {P.figures}")


if __name__ == "__main__":
    main()
