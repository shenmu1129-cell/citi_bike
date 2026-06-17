from __future__ import annotations

import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .regional_common import PATHS, ensure_dirs, log, regression_metrics


TARGET = "net_flow_next_hour"
NON_FEATURES = {
    "datetime",
    "grid_id",
    "pickup_count_next_hour",
    "dropoff_count_next_hour",
    "net_flow_next_hour",
}


def split_by_time(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    times = np.array(sorted(pd.to_datetime(df["datetime"]).unique()))
    train_end = int(len(times) * train_ratio)
    val_end = int(len(times) * (train_ratio + val_ratio))
    train = df[pd.to_datetime(df["datetime"]) < times[train_end]].copy()
    val = df[(pd.to_datetime(df["datetime"]) >= times[train_end]) & (pd.to_datetime(df["datetime"]) < times[val_end])].copy()
    test = df[pd.to_datetime(df["datetime"]) >= times[val_end]].copy()
    return train, val, test


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]


def train_baselines(config: Dict, quick: bool = False) -> pd.DataFrame:
    ensure_dirs()
    df = pd.read_csv(PATHS["processed"] / "regional_model_dataset.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    features = feature_columns(df)
    train, val, test = split_by_time(df, float(config["train_ratio"]), float(config["val_ratio"]))
    fit_train = train
    max_rows = 120_000 if quick else 450_000
    if len(fit_train) > max_rows:
        idx = np.linspace(0, len(fit_train) - 1, max_rows, dtype=int)
        fit_train = fit_train.iloc[idx]
        log(f"baseline training rows capped to {len(fit_train):,}")

    X_train = fit_train[features].to_numpy()
    y_train = fit_train[TARGET].to_numpy()
    X_test = test[features].to_numpy()
    y_test = test[TARGET].to_numpy()
    models = {
        "Ridge": (
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
            "linear baseline",
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=50 if quick else 120,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=int(config["random_state"]),
                n_jobs=-1,
            ),
            "nonlinear tree baseline",
        ),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(
                max_iter=80 if quick else 180,
                learning_rate=0.06,
                l2_regularization=0.01,
                random_state=int(config["random_state"]),
            ),
            "strong structured-data baseline",
        ),
        "MLP": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(96, 48),
                            max_iter=80 if quick else 160,
                            early_stopping=True,
                            random_state=int(config["random_state"]),
                        ),
                    ),
                ]
            ),
            "tabular neural-network baseline",
        ),
    }
    rows = []
    for name, (model, note) in models.items():
        log(f"fit regional baseline: {name}")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({"model": name, **regression_metrics(y_test, pred), "notes": note})
        with (PATHS["models"] / f"regional_baseline_{name.lower().replace(' ', '_')}.pkl").open("wb") as f:
            pickle.dump({"model": model, "features": features}, f)
    metrics = pd.DataFrame(rows)
    metrics_path = PATHS["tables"] / "regional_model_metrics.csv"
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        keep = existing[~existing["model"].isin(metrics["model"])]
        if not keep.empty:
            metrics = pd.concat([metrics, keep], ignore_index=True)
    metrics = metrics.sort_values("RMSE")
    metrics.to_csv(PATHS["tables"] / "regional_model_metrics.csv", index=False)
    log("saved regional baseline metrics")
    return metrics
