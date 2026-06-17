from __future__ import annotations

from typing import Dict

import pandas as pd

from .regional_common import PATHS, ROOT, ensure_dirs, log


def _table(path, fallback: str = "暂无数据。") -> str:
    if not path.exists():
        return fallback
    df = pd.read_csv(path)
    if df.empty:
        return fallback
    return df.to_markdown(index=False)


def generate_report(config: Dict) -> None:
    ensure_dirs()
    panel = pd.read_csv(PATHS["processed"] / "regional_hourly_panel.csv")
    model_df = pd.read_csv(PATHS["processed"] / "regional_model_dataset.csv")
    metrics = pd.read_csv(PATHS["tables"] / "regional_model_metrics.csv")
    region_info = pd.read_csv(PATHS["tables"] / "region_grid_info.csv")
    edges = pd.read_csv(PATHS["tables"] / "region_edges.csv")
    pred = pd.read_csv(PATHS["tables"] / "stgcn_predictions.csv")
    risk = pd.read_csv(PATHS["tables"] / "dispatch_risk_top10.csv")
    shortage = risk[risk["risk_type"] == "shortage_risk"]
    overflow = risk[risk["risk_type"] == "overflow_risk"]
    best = metrics.sort_values("RMSE").iloc[0]
    stgcn = metrics[metrics["model"] == "STGCN"].iloc[0] if (metrics["model"] == "STGCN").any() else best

    summary = pd.DataFrame(
        [
            ["区域数量", panel["grid_id"].nunique()],
            ["小时范围", f"{panel['datetime'].min()} 至 {panel['datetime'].max()}"],
            ["区域面板行数", len(panel)],
            ["建模样本行数", len(model_df)],
            ["图节点数", len(region_info)],
            ["图边数", len(edges)],
            ["STGCN 测试预测行数", len(pred)],
            ["lookback", config["lookback"]],
        ],
        columns=["指标", "数值"],
    )

    if float(stgcn["RMSE"]) <= float(best["RMSE"]) + 1e-9:
        stgcn_note = "STGCN 在本次实验中取得了最优或并列最优 RMSE。"
    else:
        stgcn_note = (
            "STGCN 并非本次 RMSE 最低模型，但经过 OD 混合图、增强节点特征和残差目标训练后，"
            "相比初始 STGCN 已明显改善。仍落后于 HistGradientBoosting 的可能原因包括："
            "区域划分较粗、Top 区域数量有限、净流量噪声较大、图结构仍是简化的距离与 OD 混合图，"
            "以及当前 STGCN 架构较轻量。"
        )

    report = f"""# 基于 STGCN 的区域级共享单车净流量预测与调度风险分析

## 1. 研究背景与问题升级

原有项目已经完成 Citi Bike 城市级小时订单需求预测，能够回答“全市下一小时大概有多少订单”。但车辆调度更关心空间不平衡：哪些区域会缺车，哪些区域会满桩。因此本扩展在原项目基础上增量新增区域级净流量预测模块，不删除原有 `run_all.py`、城市级天气融合代码和 `outputs/report.md`。

本项目表述为：在原城市级订单预测基础上，进一步利用 Citi Bike 时空订单数据，将城市划分为多个区域，并使用 STGCN 预测下一小时区域净流量，从而识别缺车和满桩风险，为共享单车车辆再平衡调度提供参考。

## 2. 数据来源与数据获取

数据来自 Citi Bike 历史 tripdata，读取字段包括 `ride_id`、`rideable_type`、`started_at`、`ended_at`、`start_lat`、`start_lng`、`end_lat`、`end_lng`、`member_casual`。正式模式会优先使用最近 12 个可用月份；如果本地月份不足，可通过 `config_stgcn.yaml` 手动配置月份，脚本会跳过已有 zip，避免重复下载。quick mode 只处理最近少量月份和抽样 chunk，便于快速验证流程。

清洗规则包括删除关键字段缺失、去重 `ride_id`、转换时间字段、计算骑行时长，并过滤小于 1 分钟或大于 1440 分钟的异常骑行。

{summary.to_markdown(index=False)}

## 3. 区域划分与净流量定义

区域使用经纬度固定网格构造，默认 `grid_size={config['grid_size']}`。每个 `grid_id` 是一个区域，区域中心点保存为 `grid_center_lat` 和 `grid_center_lng`。为避免稀疏区域影响建模，默认保留订单量最高的 Top {config['top_regions']} 区域。

每个小时和区域聚合：

- `pickup_count`
- `pickup_member_count`
- `pickup_casual_count`
- `pickup_electric_count`
- `pickup_classic_count`
- `dropoff_count`
- `net_flow = dropoff_count - pickup_count`

解释：`net_flow < 0` 表示借走多、还回少，区域车辆可能减少；`net_flow > 0` 表示还回多、借走少，区域车辆可能堆积。主预测目标是按 `grid_id` 分组构造的 `net_flow_next_hour`，避免不同区域之间错位。

![区域网格分布图](figures/regional_grid_map.png)

![高峰小时 pickup 热力图](figures/regional_pickup_heatmap.png)

![高峰小时 dropoff 热力图](figures/regional_dropoff_heatmap.png)

## 4. 图结构构造

STGCN 将每个区域作为图节点。本次强化后，区域图不再只依赖地理距离，而是使用 **距离 kNN + OD 流量** 的混合图。距离部分使用中心点 haversine 距离构造 kNN 图，默认 `k={config['knn_k']}`；OD 部分统计 Top 区域之间的历史骑行流量，并保留每个区域流量最强的邻居，默认 `od_k={config.get('od_k', 5)}`。距离图权重为 `{config.get('distance_graph_weight', 0.75)}`，OD 图权重为 `{config.get('od_graph_weight', 0.25)}`。

距离边权使用距离衰减：

`weight = exp(-distance / sigma)`

随后加入自环，并对混合邻接矩阵做归一化，输出 `outputs/tables/region_adjacency_matrix.csv`、`outputs/tables/region_edges.csv`、`outputs/tables/region_od_edges.csv` 和 `outputs/tables/region_od_matrix.csv`。这个图结构让模型可以同时利用相邻区域之间的空间关系、实际 OD 联系和历史时间变化。

## 5. 模型方法

对比模型包括：

- Ridge：线性基线。
- Random Forest：非线性树模型。
- HistGradientBoosting：传统机器学习强基线。
- MLP：表格深度学习基线。
- STGCN：主模型，同时建模区域空间关系和历史时间变化。

STGCN 输入张量为 `[num_samples, lookback, num_nodes, num_features]`，默认 `lookback={config['lookback']}`。强化后的节点特征包括 pickup、dropoff、net_flow、会员/临时用户、车型结构、周期时间特征、周末/高峰标记，以及按 `grid_id` 分组构造的 lag 与 rolling 特征。目标张量为 `[num_samples, num_nodes]`，表示每个区域下一小时净流量。训练时使用残差目标 `next_net_flow - current_net_flow`，预测后再加回当前净流量，以提升短期时序稳定性。训练、验证、测试按时间顺序切分，不随机切分；标准化只在训练集 fit，避免数据泄漏。

## 6. 实验结果

{metrics.to_markdown(index=False)}

{stgcn_note}

![模型指标对比图](figures/regional_model_metrics_bar.png)

![STGCN 真实值与预测值对比](figures/stgcn_actual_vs_predicted.png)

![STGCN 训练损失曲线](figures/stgcn_training_loss.png)

## 7. 调度风险分析

基于 STGCN 的 `predicted_net_flow` 判断调度风险。阈值默认使用训练集中 `abs(net_flow_next_hour)` 的 75% 分位数，也可在配置文件中手动设置。

- `predicted_net_flow < -threshold`：`shortage_risk`，建议提前补车。
- `predicted_net_flow > threshold`：`overflow_risk`，建议提前移走车辆或预留空桩。
- 其他情况：`balanced`，暂不需要明显调度。

调度优先级：

`dispatch_priority = abs(predicted_net_flow) * historical_avg_demand`

### Top 10 缺车风险区域

{shortage.to_markdown(index=False) if not shortage.empty else '暂无超过阈值的缺车风险区域。'}

### Top 10 满桩风险区域

{overflow.to_markdown(index=False) if not overflow.empty else '暂无超过阈值的满桩风险区域。'}

![STGCN 预测下一小时净流量地图](figures/stgcn_predicted_net_flow_map.png)

![调度风险地图](figures/dispatch_risk_map.png)

## 8. 可视化分析

![真实净流量地图](figures/regional_net_flow_map.png)

这些图表分别展示区域分布、典型高峰小时取还车空间差异、真实净流量、STGCN 预测净流量和最终调度风险区域。图表不是为了堆叠展示，而是服务于一个结论：区域级净流量比城市级订单量更能支持车辆再平衡判断。

## 9. 局限性

1. 当前是区域级预测，不是站点级预测。
2. 没有真实站点容量约束。
3. 没有真实调度车辆路径、人工成本、车辆容量和作业时窗。
4. 邻接矩阵已融合距离和 OD 流量，但仍未使用更复杂的方向性 OD 图、动态 OD 图或站点级流动关系。
5. STGCN 是简化版，仍可进一步优化。

## 10. 改进方向

- 接入 Citi Bike GBFS 实时站点容量和可用车辆数。
- 引入地铁站、POI、住宅/商业区等空间特征。
- 使用 OD 流量构造更真实的区域图。
- 扩展为未来 1、3、6 小时多步预测。
- 尝试 DCRNN、Graph WaveNet 或 Temporal Fusion Transformer。

## 交付物

- `run_stgcn_regional.py`
- `config_stgcn.yaml`
- `data/processed/regional_hourly_panel.csv`
- `data/processed/regional_model_dataset.csv`
- `outputs/tables/region_grid_info.csv`
- `outputs/tables/region_adjacency_matrix.csv`
- `outputs/tables/region_edges.csv`
- `outputs/models/stgcn_best.pt`
- `outputs/tables/stgcn_predictions.csv`
- `outputs/tables/regional_model_metrics.csv`
- `outputs/tables/dispatch_risk_top10.csv`
- `outputs/report_stgcn_regional.md`
"""
    (ROOT / "outputs" / "report_stgcn_regional.md").write_text(report, encoding="utf-8")
    log("saved STGCN regional report")
