from __future__ import annotations

import os
from typing import Dict, Iterable, List

from .regional_common import ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance

from .regional_common import PATHS, ensure_dirs, log, regression_metrics
from .train_baselines_regional import NON_FEATURES, TARGET, feature_columns, split_by_time


SUBWAY_FEATURES = [
    "nearest_subway_distance",
    "subway_count_500m",
    "subway_count_1000m",
    "transit_congestion_index",
]

GEO_FEATURES = ["grid_center_lat", "grid_center_lng"]

COUNT_CURRENT_FEATURES = [
    "pickup_count",
    "pickup_member_count",
    "pickup_casual_count",
    "pickup_electric_count",
    "pickup_classic_count",
    "dropoff_count",
    "net_flow",
]

TIME_FEATURES = [
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "is_rush_hour",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

HISTORY_FEATURES = [
    "pickup_lag_1",
    "pickup_lag_2",
    "pickup_lag_3",
    "pickup_lag_24",
    "pickup_lag_168",
    "dropoff_lag_1",
    "net_flow_lag_1",
    "dropoff_lag_24",
    "net_flow_lag_24",
    "dropoff_lag_168",
    "net_flow_lag_168",
    "pickup_rolling_mean_3",
    "pickup_rolling_mean_24",
    "pickup_rolling_mean_168",
    "dropoff_rolling_mean_24",
    "net_flow_rolling_mean_24",
    "net_flow_rolling_std_24",
]


def _available(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c in df.columns and c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]


def _subsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    idx = np.linspace(0, len(df) - 1, max_rows, dtype=int)
    return df.iloc[idx].copy()


def _fit_predict(model, train: pd.DataFrame, test: pd.DataFrame, features: List[str]):
    model.fit(train[features].to_numpy(), train[TARGET].to_numpy())
    pred = model.predict(test[features].to_numpy())
    return pred


def _hgb(random_state: int, quick: bool):
    return HistGradientBoostingRegressor(
        max_iter=80 if quick else 180,
        learning_rate=0.06,
        l2_regularization=0.01,
        random_state=random_state,
    )


def _rf(random_state: int, quick: bool):
    return RandomForestRegressor(
        n_estimators=60 if quick else 120,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=1,
    )


def _plot_ablation(ablation: pd.DataFrame) -> None:
    plt.figure(figsize=(9.5, 4.8))
    sns.barplot(data=ablation, x="feature_set", y="RMSE", hue="model")
    plt.xticks(rotation=25, ha="right")
    plt.title("Spatial feature ablation: RMSE")
    plt.tight_layout()
    plt.savefig(PATHS["figures"] / "spatial_feature_ablation_rmse.png", bbox_inches="tight")
    plt.close()


def _plot_importance(importance: pd.DataFrame) -> None:
    top = importance.sort_values(["source", "rank"]).groupby("source").head(15).copy()
    top["label"] = top["rank"].astype(str).str.zfill(2) + ". " + top["feature"]
    top["normalized_importance"] = top["importance"] / top.groupby("source")["importance"].transform("max").replace(0, 1)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top, y="label", x="normalized_importance", hue="source")
    plt.title("Top feature importance by method (normalized within method)")
    plt.ylabel("feature")
    plt.xlabel("normalized importance")
    plt.tight_layout()
    plt.savefig(PATHS["figures"] / "spatial_feature_importance.png", bbox_inches="tight")
    plt.close()


def _write_report(ablation: pd.DataFrame, importance: pd.DataFrame) -> None:
    subway_rows = importance[importance["feature"].isin(SUBWAY_FEATURES)].copy()
    subway_rank = (
        subway_rows.sort_values("importance", ascending=False)[["feature", "rank", "importance", "source"]].to_markdown(index=False)
        if not subway_rows.empty
        else "本次数据中没有可用地铁站特征。"
    )
    full = ablation[ablation["feature_set"] == "full"].sort_values("RMSE").head(1)
    no_subway = ablation[ablation["feature_set"] == "no_subway"].sort_values("RMSE").head(1)
    if not full.empty and not no_subway.empty:
        rmse_delta = float(no_subway.iloc[0]["RMSE"] - full.iloc[0]["RMSE"])
        decision = (
            "地铁站特征带来了可观收益，建议保留。"
            if rmse_delta > 0.25
            else "去掉地铁站特征后指标变化很小，建议不要把地铁站作为核心结论，只保留为可选空间特征。"
        )
    else:
        rmse_delta = float("nan")
        decision = "缺少 full/no_subway 对比，无法给出保留判断。"

    report = f"""# 地图空间特征有效性分析

## 结论

本分析用时间顺序切分的测试集比较不同特征组，重点检验地铁站特征是否真的影响 Citi Bike 区域净流量预测。

- `no_subway - full` 的最佳 RMSE 差值为 `{rmse_delta:.4f}`。
- 判断：{decision}

## 特征组消融实验

{ablation.sort_values(["model", "RMSE"]).to_markdown(index=False)}

![空间特征消融 RMSE](../figures/spatial_feature_ablation_rmse.png)

## 特征重要性

重要性来源包括：

- Random Forest impurity importance。
- HistGradientBoosting permutation importance。

{importance.sort_values("importance", ascending=False).head(30).to_markdown(index=False)}

![空间特征重要性](../figures/spatial_feature_importance.png)

## 地铁站特征排名

{subway_rank}

## 解释

如果地铁站特征排名靠后，或去掉地铁站后 RMSE 几乎不变，说明“地铁站位置”在当前网格粒度和目标定义下不是主导因素。更可能起作用的是当前小时供需、短期 lag、24 小时周期和区域历史均值。这种情况下，汇报里不应把地铁站说成核心驱动，而应强调用算法筛选空间特征，地铁只是候选变量之一。
"""
    (PATHS["tables"] / "spatial_feature_analysis.md").write_text(report, encoding="utf-8")


def analyze_spatial_feature_importance(config: Dict, quick: bool = False) -> pd.DataFrame:
    ensure_dirs()
    df = pd.read_csv(PATHS["processed"] / "regional_model_dataset.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    all_features = feature_columns(df)
    train, _, test = split_by_time(df, float(config["train_ratio"]), float(config["val_ratio"]))
    train_fit = _subsample(train, 120_000 if quick else 260_000)
    test_eval = _subsample(test, 40_000 if quick else 80_000)
    random_state = int(config["random_state"])

    subway = _available(df, SUBWAY_FEATURES)
    geo = _available(df, GEO_FEATURES)
    current = _available(df, COUNT_CURRENT_FEATURES)
    time_features = _available(df, TIME_FEATURES)
    history = _available(df, HISTORY_FEATURES)
    temporal_core = current + time_features + history

    feature_sets = {
        "full": all_features,
        "no_subway": [c for c in all_features if c not in subway],
        "temporal_only": temporal_core,
        "temporal_plus_geo": list(dict.fromkeys(temporal_core + geo)),
        "temporal_plus_subway": list(dict.fromkeys(temporal_core + subway)),
        "map_only": list(dict.fromkeys(geo + subway)),
    }
    rows = []
    models = {
        "HistGradientBoosting": _hgb(random_state, quick),
        "Random Forest": _rf(random_state, quick),
    }
    for set_name, features in feature_sets.items():
        if not features:
            continue
        for model_name, model in models.items():
            log(f"feature ablation: {model_name} / {set_name} ({len(features)} features)")
            pred = _fit_predict(model, train_fit, test_eval, features)
            rows.append(
                {
                    "model": model_name,
                    "feature_set": set_name,
                    "num_features": len(features),
                    **regression_metrics(test_eval[TARGET], pred),
                }
            )
    ablation = pd.DataFrame(rows).sort_values(["model", "RMSE"])
    ablation.to_csv(PATHS["tables"] / "spatial_feature_ablation.csv", index=False)

    log("fit feature-importance Random Forest")
    rf_features = feature_sets["full"]
    rf_model = _rf(random_state, quick)
    rf_model.fit(train_fit[rf_features].to_numpy(), train_fit[TARGET].to_numpy())
    rf_importance = pd.DataFrame(
        {
            "feature": rf_features,
            "importance": rf_model.feature_importances_,
            "source": "random_forest_impurity",
        }
    )

    log("fit feature-importance HistGradientBoosting")
    hgb_model = _hgb(random_state, quick)
    hgb_model.fit(train_fit[rf_features].to_numpy(), train_fit[TARGET].to_numpy())
    perm_eval = _subsample(test_eval, 12_000 if quick else 20_000)
    perm = permutation_importance(
        hgb_model,
        perm_eval[rf_features].to_numpy(),
        perm_eval[TARGET].to_numpy(),
        n_repeats=3 if quick else 5,
        random_state=random_state,
        n_jobs=1,
        scoring="neg_root_mean_squared_error",
    )
    perm_importance = pd.DataFrame(
        {
            "feature": rf_features,
            "importance": perm.importances_mean,
            "source": "hgb_permutation_rmse",
        }
    )
    importance = pd.concat([rf_importance, perm_importance], ignore_index=True)
    importance["importance"] = importance["importance"].clip(lower=0)
    importance["rank"] = importance.groupby("source")["importance"].rank(method="min", ascending=False).astype(int)
    importance = importance.sort_values(["source", "rank", "feature"])
    importance.to_csv(PATHS["tables"] / "spatial_feature_importance.csv", index=False)

    _plot_ablation(ablation)
    _plot_importance(importance)
    _write_report(ablation, importance)
    log("saved spatial feature analysis")
    return ablation
