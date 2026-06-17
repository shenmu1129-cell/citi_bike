# 基于 STGCN 的区域级共享单车净流量预测与调度风险分析

## 1. 研究背景与问题升级

原有项目已经完成 Citi Bike 城市级小时订单需求预测，能够回答“全市下一小时大概有多少订单”。但车辆调度更关心空间不平衡：哪些区域会缺车，哪些区域会满桩。因此本扩展在原项目基础上增量新增区域级净流量预测模块，不删除原有 `run_all.py`、城市级天气融合代码和 `outputs/report.md`。

本项目表述为：在原城市级订单预测基础上，进一步利用 Citi Bike 时空订单数据，将城市划分为多个区域，并使用 STGCN 预测下一小时区域净流量，从而识别缺车和满桩风险，为共享单车车辆再平衡调度提供参考。

## 2. 数据来源与数据获取

数据来自 Citi Bike 历史 tripdata，读取字段包括 `ride_id`、`rideable_type`、`started_at`、`ended_at`、`start_lat`、`start_lng`、`end_lat`、`end_lng`、`member_casual`。正式模式会优先使用最近 12 个可用月份；如果本地月份不足，可通过 `config_stgcn.yaml` 手动配置月份，脚本会跳过已有 zip，避免重复下载。quick mode 只处理最近少量月份和抽样 chunk，便于快速验证流程。

清洗规则包括删除关键字段缺失、去重 `ride_id`、转换时间字段、计算骑行时长，并过滤小于 1 分钟或大于 1440 分钟的异常骑行。

| 指标           | 数值                                        |
|:-------------|:------------------------------------------|
| 区域数量         | 40                                        |
| 小时范围         | 2025-05-31 04:00:00 至 2026-05-31 23:00:00 |
| 区域面板行数       | 351200                                    |
| 建模样本行数       | 344440                                    |
| 图节点数         | 40                                        |
| 图边数          | 243                                       |
| STGCN 测试预测行数 | 52560                                     |
| lookback     | 24                                        |

## 3. 区域划分与净流量定义

区域使用经纬度固定网格构造，默认 `grid_size=0.01`。每个 `grid_id` 是一个区域，区域中心点保存为 `grid_center_lat` 和 `grid_center_lng`。为避免稀疏区域影响建模，默认保留订单量最高的 Top 40 区域。

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

STGCN 将每个区域作为图节点。本次强化后，区域图不再只依赖地理距离，而是使用 **距离 kNN + OD 流量** 的混合图。距离部分使用中心点 haversine 距离构造 kNN 图，默认 `k=5`；OD 部分统计 Top 区域之间的历史骑行流量，并保留每个区域流量最强的邻居，默认 `od_k=5`。距离图权重为 `0.75`，OD 图权重为 `0.25`。

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

STGCN 输入张量为 `[num_samples, lookback, num_nodes, num_features]`，默认 `lookback=24`。强化后的节点特征包括 pickup、dropoff、net_flow、会员/临时用户、车型结构、周期时间特征、周末/高峰标记，以及按 `grid_id` 分组构造的 lag 与 rolling 特征。目标张量为 `[num_samples, num_nodes]`，表示每个区域下一小时净流量。训练时使用残差目标 `next_net_flow - current_net_flow`，预测后再加回当前净流量，以提升短期时序稳定性。训练、验证、测试按时间顺序切分，不随机切分；标准化只在训练集 fit，避免数据泄漏。

## 6. 实验结果

| model                |     MAE |    RMSE |       R2 |    MAPE |   sMAPE | notes                                |
|:---------------------|--------:|--------:|---------:|--------:|--------:|:-------------------------------------|
| STGCN                | 11.2189 | 16.6052 | 0.859614 | 136.059 | 109.083 | spatio-temporal graph neural network |
| MLP                  | 11.7068 | 17.2624 | 0.847679 | 142.1   | 111.628 | tabular neural-network baseline      |
| Random Forest        | 11.8308 | 18.2712 | 0.829355 | 121.423 | 114.277 | nonlinear tree baseline              |
| HistGradientBoosting | 12.3485 | 18.5617 | 0.823886 | 130.551 | 117.585 | strong structured-data baseline      |
| Ridge                | 17.1956 | 31.063  | 0.506774 | 176.577 | 123.825 | linear baseline                      |

STGCN 在本次实验中取得了最优或并列最优 RMSE。

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

| datetime            | grid_id      |   grid_center_lat |   grid_center_lng |   predicted_net_flow | risk_type     |   dispatch_priority | suggested_action   |
|:--------------------|:-------------|------------------:|------------------:|---------------------:|:--------------|--------------------:|:-------------------|
| 2026-04-14 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -600.013 | shortage_risk |            109404   | 建议提前补车             |
| 2026-05-26 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -584.848 | shortage_risk |            106639   | 建议提前补车             |
| 2026-05-19 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -566.042 | shortage_risk |            103210   | 建议提前补车             |
| 2026-05-26 17:00:00 | 40.74_-73.99 |             40.74 |            -73.99 |             -350.703 | shortage_risk |             99772.1 | 建议提前补车             |
| 2026-05-27 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -541.458 | shortage_risk |             98727.5 | 建议提前补车             |
| 2026-05-05 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -540.837 | shortage_risk |             98614.4 | 建议提前补车             |
| 2026-05-20 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -533.095 | shortage_risk |             97202.8 | 建议提前补车             |
| 2026-05-12 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -528.936 | shortage_risk |             96444.4 | 建议提前补车             |
| 2026-05-18 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -528.537 | shortage_risk |             96371.6 | 建议提前补车             |
| 2026-05-04 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -513.966 | shortage_risk |             93714.8 | 建议提前补车             |

### Top 10 满桩风险区域

| datetime            | grid_id      |   grid_center_lat |   grid_center_lng |   predicted_net_flow | risk_type     |   dispatch_priority | suggested_action   |
|:--------------------|:-------------|------------------:|------------------:|---------------------:|:--------------|--------------------:|:-------------------|
| 2026-04-15 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              512.402 | overflow_risk |             93429.6 | 建议提前移走车辆或预留空桩      |
| 2026-05-19 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              502.484 | overflow_risk |             91621.1 | 建议提前移走车辆或预留空桩      |
| 2026-05-20 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              496.221 | overflow_risk |             90479.3 | 建议提前移走车辆或预留空桩      |
| 2026-04-14 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              493.285 | overflow_risk |             89943.9 | 建议提前移走车辆或预留空桩      |
| 2026-04-14 08:00:00 | 40.74_-73.99 |             40.74 |            -73.99 |              316.002 | overflow_risk |             89900   | 建议提前移走车辆或预留空桩      |
| 2026-05-06 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              492.328 | overflow_risk |             89769.4 | 建议提前移走车辆或预留空桩      |
| 2026-04-28 08:00:00 | 40.74_-73.99 |             40.74 |            -73.99 |              306.929 | overflow_risk |             87318.8 | 建议提前移走车辆或预留空桩      |
| 2026-05-28 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              473.437 | overflow_risk |             86324.8 | 建议提前移走车辆或预留空桩      |
| 2026-05-05 08:00:00 | 40.74_-73.99 |             40.74 |            -73.99 |              303.198 | overflow_risk |             86257.2 | 建议提前移走车辆或预留空桩      |
| 2026-04-29 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              471.924 | overflow_risk |             86049   | 建议提前移走车辆或预留空桩      |

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
