#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SKILL_SCRIPTS = (
    Path("/Users/shenmu/Downloads/同步空间/Mac/CodexWorkspace/skill")
    / "scipilot-figure-skill"
    / "scripts"
)
if SKILL_SCRIPTS.exists():
    sys.path.insert(0, str(SKILL_SCRIPTS))

try:
    from export_figure import export_figure
    from check_figure import check_figure
except Exception:
    export_figure = None
    check_figure = None


FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
REPORT_PATH = ROOT / "outputs" / "report_stgcn_presentation.md"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Palette adapted from the user's second reference image.
COLORS = {
    "taupe": "#948080",
    "peach": "#e99e80",
    "blue": "#0c6db8",
    "cyan": "#28c0cd",
    "orange": "#e06a36",
    "teal_deep": "#557f7c",
    "teal_soft": "#a1c4bf",
    "cream": "#ebc9b9",
    "red": "#cc4c36",
    "gray": "#b6b6b6",
    "mint": "#61b2ac",
    "aqua_soft": "#9ed0cf",
    "pink_soft": "#efc9c8",
    "salmon": "#ec9c9d",
    "rose": "#e78081",
    "ink": "#1d2433",
    "grid": "#d8dee5",
}


def set_style() -> None:
    sns.set_theme(
        context="talk",
        style="whitegrid",
        rc={
            "font.family": "DejaVu Sans",
            "axes.labelsize": 12,
            "axes.titlesize": 15,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.edgecolor": COLORS["grid"],
            "grid.color": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        },
    )


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)


def save_figure(fig: plt.Figure, basename: str, size: tuple[float, float]) -> list[Path]:
    out_base = FIG_DIR / basename
    if export_figure is not None:
        paths = export_figure(
            fig,
            str(out_base),
            formats=["png", "svg", "pdf"],
            size_inches=size,
            dpi=300,
            grayscale_preview=True,
            pad_inches=0.05,
        )
    else:
        fig.set_size_inches(*size)
        paths = []
        for ext in ("png", "svg", "pdf"):
            path = f"{out_base}.{ext}"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=300)
            paths.append(path)
    plt.close(fig)
    return [Path(p) for p in paths]


def clean_old_generated_figures() -> None:
    patterns = [
        "ppt_fig_*",
        "presentation_fig1_model_performance*",
        "presentation_fig2_feature_evidence*",
        "presentation_fig3_stgcn_diagnostics*",
        "presentation_fig4_dispatch_risk*",
    ]
    for pattern in patterns:
        for path in FIG_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def model_rmse(metrics: pd.DataFrame) -> None:
    plot_df = metrics.sort_values("RMSE")
    palette = [COLORS[c] for c in ["blue", "cyan", "taupe", "peach", "orange"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    sns.barplot(data=plot_df, y="model", x="RMSE", hue="model", palette=palette, legend=False, ax=ax)
    ax.set_title("STGCN achieves the lowest regional net-flow RMSE")
    ax.set_xlabel("RMSE (bikes/hour)")
    ax.set_ylabel("")
    ax.bar_label(ax.containers[0], fmt="%.2f", padding=4, fontsize=10)
    save_figure(fig, "ppt_fig_01_model_rmse", (8.2, 4.6))


def model_r2(metrics: pd.DataFrame) -> None:
    plot_df = metrics.sort_values("R2", ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.hlines(plot_df["model"], 0, plot_df["R2"], color=COLORS["grid"], linewidth=3)
    ax.scatter(plot_df["R2"], plot_df["model"], s=190, color=COLORS["cyan"], edgecolor=COLORS["blue"], linewidth=1.8)
    for _, row in plot_df.iterrows():
        ax.text(row["R2"] + 0.008, row["model"], f"{row['R2']:.3f}", va="center", fontsize=11)
    ax.set_xlim(0.45, 0.91)
    ax.set_title("STGCN explains the most regional variance")
    ax.set_xlabel("$R^2$")
    ax.set_ylabel("")
    save_figure(fig, "ppt_fig_02_model_r2", (8.2, 4.6))


def feature_ablation(ablation: pd.DataFrame) -> None:
    feature_order = ["full", "temporal_plus_subway", "no_subway", "temporal_plus_geo", "temporal_only", "map_only"]
    labels = {
        "full": "Full",
        "temporal_plus_subway": "+ Subway",
        "no_subway": "No subway",
        "temporal_plus_geo": "+ Geo",
        "temporal_only": "Temporal only",
        "map_only": "Map only",
    }
    plot_df = ablation.copy()
    plot_df["feature_set"] = pd.Categorical(plot_df["feature_set"], feature_order, ordered=True)
    plot_df["feature_label"] = plot_df["feature_set"].astype(str).map(labels)
    plot_df = plot_df.sort_values(["feature_set", "model"])

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    sns.lineplot(
        data=plot_df,
        x="feature_label",
        y="RMSE",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
        palette=[COLORS["blue"], COLORS["orange"]],
        linewidth=2.2,
        markersize=9,
        ax=ax,
    )
    ax.set_title("Spatial features add value, but maps alone are insufficient")
    ax.set_xlabel("")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="", frameon=False, loc="upper left")
    save_figure(fig, "ppt_fig_03_feature_ablation", (8.4, 4.8))


def feature_importance(importance: pd.DataFrame) -> None:
    plot_df = (
        importance[importance["source"].eq("hgb_permutation_rmse")]
        .sort_values("rank")
        .head(10)
        .sort_values("importance")
    )
    bar_colors = [COLORS["cyan"] if "subway" in f else COLORS["blue"] for f in plot_df["feature"]]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.barh(plot_df["feature"], plot_df["importance"], color=bar_colors)
    ax.set_title("Current demand state dominates feature importance")
    ax.set_xlabel("Permutation RMSE increase")
    ax.set_ylabel("")
    save_figure(fig, "ppt_fig_04_feature_importance", (8.2, 5.0))


def training_loss(loss: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(loss["epoch"], loss["train_loss"], color=COLORS["blue"], linewidth=2.6, label="Train")
    ax.plot(loss["epoch"], loss["val_loss"], color=COLORS["orange"], linewidth=2.6, linestyle="--", label="Validation")
    ax.set_title("STGCN training converges steadily")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend(frameon=False)
    save_figure(fig, "ppt_fig_05_training_loss", (8.2, 4.6))


def observed_predicted(pred: pd.DataFrame) -> None:
    q_low = float(pred[["y_true_net_flow", "y_pred_net_flow"]].quantile(0.01).min())
    q_high = float(pred[["y_true_net_flow", "y_pred_net_flow"]].quantile(0.99).max())
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    hb = ax.hexbin(
        pred["y_true_net_flow"],
        pred["y_pred_net_flow"],
        gridsize=55,
        cmap=sns.blend_palette([COLORS["blue"], COLORS["cyan"], COLORS["peach"]], as_cmap=True),
        bins="log",
        mincnt=1,
        extent=(q_low, q_high, q_low, q_high),
    )
    ax.plot([q_low, q_high], [q_low, q_high], color=COLORS["orange"], linestyle="--", linewidth=2)
    ax.set_xlim(q_low, q_high)
    ax.set_ylim(q_low, q_high)
    ax.set_title("Predicted net flow tracks observed net flow")
    ax.set_xlabel("Observed next-hour net flow")
    ax.set_ylabel("Predicted next-hour net flow")
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03, label="log10(count)")
    save_figure(fig, "ppt_fig_06_observed_predicted", (6.4, 5.6))


def hourly_error(pred: pd.DataFrame) -> None:
    data = pred.copy()
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["hour"] = data["datetime"].dt.hour
    hourly = data.groupby("hour", as_index=False)["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(hourly["hour"], hourly["abs_error"], color=COLORS["cyan"], marker="o", markersize=7, linewidth=2.5)
    ax.fill_between(hourly["hour"], hourly["abs_error"], color=COLORS["cyan"], alpha=0.18)
    ax.set_title("Errors rise during commute peaks")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean absolute error")
    ax.set_xticks(range(0, 24, 3))
    save_figure(fig, "ppt_fig_07_hourly_error", (8.2, 4.6))


def risk_bars(risk: pd.DataFrame, congestion: pd.DataFrame) -> None:
    risk = risk.copy()
    risk["datetime"] = pd.to_datetime(risk["datetime"])
    congestion = congestion.copy()
    congestion["datetime"] = pd.to_datetime(congestion["datetime"])

    shortage = risk[risk["risk_type"].eq("shortage_risk")].nlargest(8, "dispatch_priority").sort_values("predicted_net_flow")
    overflow = congestion.nlargest(8, "congestion_score").sort_values("congestion_score")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = shortage["datetime"].dt.strftime("%m-%d %H:%M")
    ax.barh(labels, shortage["predicted_net_flow"], color=COLORS["blue"])
    ax.axvline(0, color=COLORS["ink"], linewidth=1)
    ax.set_title(f"Top shortage risk windows: {shortage['grid_id'].iloc[0]}")
    ax.set_xlabel("Predicted net flow")
    ax.set_ylabel("")
    save_figure(fig, "ppt_fig_08_shortage_risk", (8.2, 4.8))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    labels = overflow["datetime"].dt.strftime("%m-%d %H:%M")
    ax.barh(labels, overflow["congestion_score"] / 1000.0, color=COLORS["orange"])
    ax.set_title(f"Top overflow risk windows: {overflow['grid_id'].iloc[0]}")
    ax.set_xlabel("Congestion score (thousand)")
    ax.set_ylabel("")
    save_figure(fig, "ppt_fig_09_overflow_risk", (8.2, 4.8))


def risk_locations(risk: pd.DataFrame, congestion: pd.DataFrame, region_map: pd.DataFrame) -> None:
    shortage = risk[risk["risk_type"].eq("shortage_risk")].nlargest(8, "dispatch_priority")
    overflow = congestion.nlargest(8, "congestion_score")
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(
        region_map["grid_center_lng"],
        region_map["grid_center_lat"],
        s=64,
        color="#c8d4df",
        edgecolor="white",
        linewidth=0.6,
        label="Top 40 regions",
    )
    ax.scatter(shortage["grid_center_lng"], shortage["grid_center_lat"], s=260, marker="v", color=COLORS["blue"], edgecolor=COLORS["ink"], label="Shortage")
    ax.scatter(overflow["grid_center_lng"], overflow["grid_center_lat"], s=260, marker="^", color=COLORS["orange"], edgecolor=COLORS["ink"], label="Overflow")
    ax.set_title("Highest risks concentrate around the same central grid")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(frameon=False, loc="lower left")
    save_figure(fig, "ppt_fig_10_risk_locations", (6.4, 5.6))


def dispatch_summary_without_timing(risk: pd.DataFrame, congestion: pd.DataFrame, region_map: pd.DataFrame) -> None:
    shortage = risk[risk["risk_type"].eq("shortage_risk")].nlargest(8, "dispatch_priority").copy()
    overflow = congestion.nlargest(8, "congestion_score").copy()
    shortage["datetime"] = pd.to_datetime(shortage["datetime"])
    overflow["datetime"] = pd.to_datetime(overflow["datetime"])
    shortage = shortage.sort_values("predicted_net_flow")
    overflow = overflow.sort_values("congestion_score")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})

    axes[0].barh(shortage["datetime"].dt.strftime("%m-%d %H:%M"), shortage["predicted_net_flow"], color=COLORS["blue"])
    axes[0].axvline(0, color=COLORS["ink"], linewidth=1)
    axes[0].set_title("Shortage risk")
    axes[0].set_xlabel("Predicted net flow")
    axes[0].set_ylabel(shortage["grid_id"].iloc[0])

    axes[1].barh(overflow["datetime"].dt.strftime("%m-%d %H:%M"), overflow["congestion_score"] / 1000.0, color=COLORS["orange"])
    axes[1].set_title("Overflow risk")
    axes[1].set_xlabel("Congestion score (thousand)")
    axes[1].set_ylabel(overflow["grid_id"].iloc[0])

    axes[2].scatter(region_map["grid_center_lng"], region_map["grid_center_lat"], s=52, color="#c8d4df", edgecolor="white", linewidth=0.5, label="Top 40")
    axes[2].scatter(shortage["grid_center_lng"], shortage["grid_center_lat"], s=230, marker="v", color=COLORS["blue"], edgecolor=COLORS["ink"], label="Shortage")
    axes[2].scatter(overflow["grid_center_lng"], overflow["grid_center_lat"], s=230, marker="^", color=COLORS["orange"], edgecolor=COLORS["ink"], label="Overflow")
    axes[2].set_title("Risk locations")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    axes[2].legend(frameon=False, loc="lower left")

    fig.suptitle("Predicted net flow translates into dispatch priorities", y=1.03, fontsize=18)
    fig.tight_layout()
    save_figure(fig, "presentation_fig4_dispatch_risk", (13.0, 4.4))


def markdown_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].map(lambda x: f"{x:.3f}" if abs(x) < 1000 else f"{x:,.1f}")
    return out.to_markdown(index=False)


def build_report(metrics: pd.DataFrame, ablation: pd.DataFrame, importance: pd.DataFrame, pred: pd.DataFrame, risk: pd.DataFrame, congestion: pd.DataFrame) -> None:
    stgcn = metrics[metrics["model"].eq("STGCN")].iloc[0]
    rf = metrics[metrics["model"].eq("Random Forest")].iloc[0]
    mlp = metrics[metrics["model"].eq("MLP")].iloc[0]
    ridge = metrics[metrics["model"].eq("Ridge")].iloc[0]
    rf_gain = (rf["RMSE"] - stgcn["RMSE"]) / rf["RMSE"] * 100
    mlp_gain = (mlp["RMSE"] - stgcn["RMSE"]) / mlp["RMSE"] * 100
    ridge_gain = (ridge["RMSE"] - stgcn["RMSE"]) / ridge["RMSE"] * 100

    full_best = ablation[ablation["feature_set"].eq("full")].sort_values("RMSE").iloc[0]
    no_subway_best = ablation[ablation["feature_set"].eq("no_subway")].sort_values("RMSE").iloc[0]
    temporal_best = ablation[ablation["feature_set"].eq("temporal_only")].sort_values("RMSE").iloc[0]
    map_best = ablation[ablation["feature_set"].eq("map_only")].sort_values("RMSE").iloc[0]
    subway_delta = no_subway_best["RMSE"] - full_best["RMSE"]
    temporal_delta = temporal_best["RMSE"] - full_best["RMSE"]
    pred_dates = pd.to_datetime(pred["datetime"])
    top_shortage = risk[risk["risk_type"].eq("shortage_risk")].nlargest(1, "dispatch_priority").iloc[0]
    top_congestion = congestion.nlargest(1, "congestion_score").iloc[0]
    top_imp = importance[importance["source"].eq("hgb_permutation_rmse")].sort_values("rank").head(8)

    REPORT_PATH.write_text(
        f"""# Citi Bike 区域 STGCN 实验汇报版

> 本文件是最终保留的汇报版 Markdown。图表已按 PPT 使用场景拆成单图，不再使用 A/B/C/D 多面板标记；配色采用你提供的 5 色柱状图参考，并整体调整为科技蓝风格。

## 1. 汇报结论

- **任务升级**：从全市小时订单量预测，升级为 Top 40 区域的下一小时净流量预测；`net_flow = dropoff - pickup`，正值代表可能满桩，负值代表可能缺车。
- **模型结果**：STGCN 是本轮最优模型，RMSE = **{stgcn['RMSE']:.3f}**，R² = **{stgcn['R2']:.3f}**；相对 Random Forest 的 RMSE 降低 **{rf_gain:.1f}%**，相对 MLP 降低 **{mlp_gain:.1f}%**，相对 Ridge 降低 **{ridge_gain:.1f}%**。
- **空间特征判断**：完整特征组最佳 RMSE = **{full_best['RMSE']:.3f}**；去掉地铁特征后最佳 RMSE 上升 **{subway_delta:.3f}**，说明地铁特征不是主驱动，但有稳定增益；只用地图特征的 RMSE = **{map_best['RMSE']:.3f}**，不能替代时序需求状态。
- **调度落地**：最高缺车风险出现在 `{top_shortage['datetime']}` 的 `{top_shortage['grid_id']}`，预测净流量 **{top_shortage['predicted_net_flow']:.1f}**；最高地铁相关满桩拥堵风险出现在 `{top_congestion['datetime']}` 的 `{top_congestion['grid_id']}`，拥堵分数 **{top_congestion['congestion_score']:.1f}**。

## 2. 数据与实验设置

| 项目 | 数值 |
|:--|:--|
| 区域数量 | 40 |
| 测试预测行数 | {len(pred):,} |
| 测试时间范围 | {pred_dates.min()} 至 {pred_dates.max()} |
| 预测目标 | 下一小时区域净流量 `net_flow_next_hour` |
| 图结构 | 区域距离 kNN + 历史 OD 流量混合图 |
| STGCN 输入窗口 | 24 小时 lookback |

## 3. 模型效果

![模型 RMSE 对比](figures/ppt_fig_01_model_rmse.png)

![模型 R2 对比](figures/ppt_fig_02_model_r2.png)

{markdown_table(metrics.sort_values('RMSE')[['model', 'MAE', 'RMSE', 'R2', 'sMAPE']])}

## 4. 空间特征有效性

![特征组消融](figures/ppt_fig_03_feature_ablation.png)

![特征重要性](figures/ppt_fig_04_feature_importance.png)

- 完整特征组优于 `temporal_only`，RMSE 改善 **{temporal_delta:.3f}**。
- 去掉地铁特征后 RMSE 变差 **{subway_delta:.3f}**，因此地铁变量有价值。
- `map_only` 的 RMSE 高达 **{map_best['RMSE']:.3f}**，说明地图特征只能补充区域差异，不能替代历史供需。

{markdown_table(top_imp[['rank', 'feature', 'importance']])}

## 5. STGCN 诊断

![训练损失](figures/ppt_fig_05_training_loss.png)

![真实值与预测值](figures/ppt_fig_06_observed_predicted.png)

![分小时误差](figures/ppt_fig_07_hourly_error.png)

## 6. 风险转化

![缺车风险](figures/ppt_fig_08_shortage_risk.png)

![满桩风险](figures/ppt_fig_09_overflow_risk.png)

![风险位置](figures/ppt_fig_10_risk_locations.png)

风险定义：

- `predicted_net_flow < -threshold`：缺车风险，建议提前补车。
- `predicted_net_flow > threshold`：满桩/拥堵风险，建议提前移走车辆或预留空桩。
- 调度优先级使用 `abs(predicted_net_flow) * historical_avg_demand`，满桩拥堵分数进一步加入地铁相关拥堵指数。

## 7. PPT 图表清单

- `outputs/figures/ppt_fig_01_model_rmse.png`
- `outputs/figures/ppt_fig_02_model_r2.png`
- `outputs/figures/ppt_fig_03_feature_ablation.png`
- `outputs/figures/ppt_fig_04_feature_importance.png`
- `outputs/figures/ppt_fig_05_training_loss.png`
- `outputs/figures/ppt_fig_06_observed_predicted.png`
- `outputs/figures/ppt_fig_07_hourly_error.png`
- `outputs/figures/ppt_fig_08_shortage_risk.png`
- `outputs/figures/ppt_fig_09_overflow_risk.png`
- `outputs/figures/ppt_fig_10_risk_locations.png`
""",
        encoding="utf-8",
    )


def audit_outputs() -> None:
    if check_figure is None:
        return
    for path in FIG_DIR.glob("ppt_fig_*.png"):
        issues, _ = check_figure(str(path), min_dpi=300)
        fails = [msg for severity, msg in issues if severity == "FAIL"]
        if fails:
            raise RuntimeError(f"Figure audit failed for {path}: {fails}")


def main() -> None:
    set_style()
    clean_old_generated_figures()

    metrics = read_table("regional_model_metrics.csv")
    ablation = read_table("spatial_feature_ablation.csv")
    importance = read_table("spatial_feature_importance.csv")
    pred = read_table("stgcn_predictions.csv")
    loss = read_table("stgcn_training_loss.csv")
    risk = read_table("dispatch_risk_top10.csv")
    congestion = read_table("congestion_risk_top10.csv")
    region_map = read_table("region_map_features.csv")

    model_rmse(metrics)
    model_r2(metrics)
    feature_ablation(ablation)
    feature_importance(importance)
    training_loss(loss)
    observed_predicted(pred)
    hourly_error(pred)
    risk_bars(risk, congestion)
    risk_locations(risk, congestion, region_map)
    dispatch_summary_without_timing(risk, congestion, region_map)
    audit_outputs()
    build_report(metrics, ablation, importance, pred, risk, congestion)

    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote PPT-ready figures under {FIG_DIR}")


if __name__ == "__main__":
    main()
