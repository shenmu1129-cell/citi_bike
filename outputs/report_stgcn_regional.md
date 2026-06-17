# Citi Bike 地图相关区域拥堵与调度风险分析

## 1. 研究背景与问题升级

原有项目已经完成 Citi Bike 城市级小时订单需求预测，能够回答“全市下一小时大概有多少订单”。但调度问题真正关心的是空间不平衡：某个区域下一小时会不会还车过多导致满桩拥堵，或者借车过多导致缺车。因此本扩展把项目主线调整为 **区域级拥堵/缺车风险识别与调度分析**，原有城市级代码保留作为历史基线，不作为本报告重点。

天气特征已从区域调度模块中剔除。原因是本任务的核心不是解释全市订单波动，而是识别“哪个地方会堆车或缺车”；这类风险更直接由起终点经纬度、区域历史流入流出、地铁站位置和空间邻接关系决定。

## 2. 数据来源与数据获取

核心数据来自 Citi Bike 历史 tripdata，读取字段包括 `ride_id`、`rideable_type`、`started_at`、`ended_at`、`start_lat`、`start_lng`、`end_lat`、`end_lng`、`member_casual`。这些字段可以直接构造每个区域每小时的借车、还车和净流量。

地图数据使用 MTA Subway Stations 开放数据。脚本优先读取 `data/raw/spatial/mta_subway_stations.csv`，如果本地不存在则尝试在线下载；如果下载失败，会使用离线降级逻辑生成默认值，不影响 Citi Bike 主流程运行。

清洗规则包括删除关键字段缺失、去重 `ride_id`、转换时间字段、计算骑行时长，并过滤小于 1 分钟或大于 1440 分钟的异常骑行。

| 指标           | 数值                                                                                       |
|:-------------|:-----------------------------------------------------------------------------------------|
| 区域数量         | 40                                                                                       |
| 小时范围         | 2025-05-31 04:00:00 至 2026-05-31 23:00:00                                                |
| 区域面板行数       | 351200                                                                                   |
| 建模样本行数       | 344440                                                                                   |
| 图节点数         | 40                                                                                       |
| 图边数          | 243                                                                                      |
| STGCN 测试预测行数 | 52560                                                                                    |
| lookback     | 24                                                                                       |
| 地图特征列        | nearest_subway_distance, subway_count_500m, subway_count_1000m, transit_congestion_index |

## 3. 区域划分、地图特征与净流量定义

区域使用经纬度固定网格构造，默认 `grid_size=0.01`。每个 `grid_id` 是一个区域，区域中心点保存为 `grid_center_lat` 和 `grid_center_lng`。为避免稀疏区域影响建模，默认保留订单量最高的 Top 40 区域。

每个小时和区域聚合：

- `pickup_count`
- `pickup_member_count`
- `pickup_casual_count`
- `pickup_electric_count`
- `pickup_classic_count`
- `dropoff_count`
- `net_flow = dropoff_count - pickup_count`

解释：`net_flow < 0` 表示借走多、还回少，区域车辆可能减少，容易缺车；`net_flow > 0` 表示还回多、借走少，区域车辆可能堆积，容易出现满桩拥堵。主预测目标是按 `grid_id` 分组构造的 `net_flow_next_hour`，避免不同区域之间错位。

地铁站地图特征包括：

- `nearest_subway_distance`：区域中心到最近地铁站距离。
- `subway_count_500m`：区域 500m 内地铁站数量。
- `subway_count_1000m`：区域 1000m 内地铁站数量。
- `transit_congestion_index`：基于地铁站密度和最近地铁距离构造的地铁相关拥堵强度指标。

这些特征的含义是：地铁站附近常出现通勤潮汐，早晚高峰更容易出现集中借车或集中还车。把这些空间变量加入模型，可以让模型区分普通高需求区域和交通枢纽型风险区域。

### 地图特征统计

|       |   nearest_subway_distance |   subway_count_500m |   subway_count_1000m |   transit_congestion_index |
|:------|--------------------------:|--------------------:|---------------------:|---------------------------:|
| count |                     40    |               40    |                40    |                      40    |
| mean  |                    358.52 |                2.7  |                 8.8  |                      12.49 |
| std   |                    227.39 |                2.62 |                 5.72 |                       8.41 |
| min   |                     69.42 |                0    |                 0    |                       0.99 |
| 25%   |                    202.21 |                1    |                 4    |                       5.93 |
| 50%   |                    315.47 |                3    |                 8    |                      10.68 |
| 75%   |                    451.08 |                4.25 |                13.25 |                      18.11 |
| max   |                   1011.36 |               13    |                21    |                      39.5  |

![区域网格分布图](figures/regional_grid_map.png)

![地铁站与 Citi Bike 区域叠加图](figures/subway_bike_grid_map.png)

![高峰小时 pickup 热力图](figures/regional_pickup_heatmap.png)

![高峰小时 dropoff 热力图](figures/regional_dropoff_heatmap.png)

## 4. 空间图结构构造

STGCN 将每个区域作为图节点。本次区域图使用 **距离 kNN + OD 流量** 的混合图。距离部分使用中心点 haversine 距离构造 kNN 图，默认 `k=5`；OD 部分统计 Top 区域之间的历史骑行流量，并保留每个区域流量最强的邻居，默认 `od_k=5`。距离图权重为 `0.75`，OD 图权重为 `0.25`。

距离边权使用距离衰减：

`weight = exp(-distance / sigma)`

随后加入自环，并对混合邻接矩阵做归一化，输出 `outputs/tables/region_adjacency_matrix.csv`、`outputs/tables/region_edges.csv`、`outputs/tables/region_od_edges.csv` 和 `outputs/tables/region_od_matrix.csv`。这个图结构让模型可以同时利用相邻区域之间的空间关系、实际 OD 联系和历史时间变化。

## 5. 模型方法与特征

对比模型包括：

- Ridge：线性基线。
- Random Forest：非线性树模型。
- HistGradientBoosting：传统机器学习强基线。
- MLP：表格深度学习基线。
- STGCN：主模型，同时建模区域空间关系和历史时间变化。

STGCN 输入张量为 `[num_samples, lookback, num_nodes, num_features]`，默认 `lookback=24`。节点特征包括 pickup、dropoff、net_flow、会员/临时用户、车型结构、周期时间特征、周末/高峰标记、lag/rolling 历史特征，以及地铁站距离和地铁站密度等地图特征。目标张量为 `[num_samples, num_nodes]`，表示每个区域下一小时净流量。训练时使用残差目标 `next_net_flow - current_net_flow`，预测后再加回当前净流量，以提升短期时序稳定性。训练、验证、测试按时间顺序切分，不随机切分；标准化只在训练集 fit，避免数据泄漏。


## 6. 空间特征有效性检验

为了避免由人工经验直接指定“地铁站一定重要”，本项目新增特征组消融和重要性分析。实验比较 `full`、`no_subway`、`temporal_only`、`temporal_plus_geo`、`temporal_plus_subway` 和 `map_only` 等特征组合，并使用 HistGradientBoosting 与 Random Forest 两类模型验证。

核心结论：去掉地铁特征后最佳 RMSE 上升 0.940，说明地铁特征不是第一驱动因素，但有稳定增益，建议保留为候选空间特征。

同时，地图特征单独预测效果很弱，说明 Citi Bike 调度风险主要仍由当前供需状态、历史净流量、小时周期和高峰时段驱动；地图特征的作用是补充解释区域差异，而不是替代时序特征。

### 特征组消融结果

| model                | feature_set          |   num_features |     MAE |    RMSE |          R2 |     MAPE |   sMAPE |
|:---------------------|:---------------------|---------------:|--------:|--------:|------------:|---------:|--------:|
| HistGradientBoosting | full                 |             39 | 12.1442 | 18.1475 | 0.831658    | 129.85   | 116.112 |
| HistGradientBoosting | temporal_plus_subway |             37 | 12.2426 | 18.3335 | 0.828189    | 130.244  | 116.654 |
| HistGradientBoosting | no_subway            |             35 | 12.3485 | 18.5617 | 0.823886    | 130.551  | 117.585 |
| HistGradientBoosting | temporal_plus_geo    |             35 | 12.3485 | 18.5617 | 0.823886    | 130.551  | 117.585 |
| HistGradientBoosting | temporal_only        |             33 | 12.7023 | 19.3795 | 0.808024    | 131.997  | 118.736 |
| HistGradientBoosting | map_only             |              6 | 21.7638 | 44.2251 | 0.000237615 |  98.4133 | 179.291 |
| Random Forest        | full                 |             39 | 11.3012 | 17.3312 | 0.846461    | 120.12   | 110.13  |
| Random Forest        | temporal_plus_subway |             37 | 11.4453 | 17.5593 | 0.842393    | 120.912  | 111.043 |
| Random Forest        | temporal_plus_geo    |             35 | 11.8019 | 18.2139 | 0.830424    | 121.487  | 114.119 |
| Random Forest        | no_subway            |             35 | 11.8308 | 18.2712 | 0.829355    | 121.423  | 114.277 |
| Random Forest        | temporal_only        |             33 | 12.1489 | 18.8448 | 0.818472    | 124.055  | 115.972 |
| Random Forest        | map_only             |              6 | 21.7628 | 44.2245 | 0.000263151 |  98.5348 | 178.906 |

![空间特征消融结果](figures/spatial_feature_ablation_rmse.png)

### 特征重要性 Top 项

| feature                 |   importance | source                 |   rank |
|:------------------------|-------------:|:-----------------------|-------:|
| net_flow                |   16.4728    | hgb_permutation_rmse   |      1 |
| hour_sin                |    7.54113   | hgb_permutation_rmse   |      2 |
| net_flow_rolling_std_24 |    5.23637   | hgb_permutation_rmse   |      3 |
| hour                    |    3.86866   | hgb_permutation_rmse   |      4 |
| net_flow_lag_168        |    2.66      | hgb_permutation_rmse   |      5 |
| is_rush_hour            |    2.15511   | hgb_permutation_rmse   |      6 |
| subway_count_1000m      |    1.31503   | hgb_permutation_rmse   |      7 |
| dropoff_count           |    1.05583   | hgb_permutation_rmse   |      8 |
| net_flow_lag_24         |    1.03774   | hgb_permutation_rmse   |      9 |
| hour_cos                |    1.01866   | hgb_permutation_rmse   |     10 |
| pickup_classic_count    |    0.859558  | hgb_permutation_rmse   |     11 |
| grid_center_lat         |    0.71399   | hgb_permutation_rmse   |     12 |
| net_flow                |    0.171147  | random_forest_impurity |      1 |
| net_flow_lag_168        |    0.122037  | random_forest_impurity |      2 |
| net_flow_lag_24         |    0.0712166 | random_forest_impurity |      3 |
| hour_sin                |    0.0529774 | random_forest_impurity |      4 |
| net_flow_rolling_std_24 |    0.0502387 | random_forest_impurity |      5 |
| hour                    |    0.0404784 | random_forest_impurity |      6 |
| net_flow_lag_1          |    0.0377291 | random_forest_impurity |      7 |
| dropoff_count           |    0.0315946 | random_forest_impurity |      8 |
| pickup_electric_count   |    0.0304992 | random_forest_impurity |      9 |
| pickup_member_count     |    0.0249218 | random_forest_impurity |     10 |
| pickup_lag_3            |    0.0245703 | random_forest_impurity |     11 |
| hour_cos                |    0.0222574 | random_forest_impurity |     12 |

![空间特征重要性](figures/spatial_feature_importance.png)


## 7. 模型预测结果

| model                |     MAE |    RMSE |       R2 |    MAPE |   sMAPE | notes                                |
|:---------------------|--------:|--------:|---------:|--------:|--------:|:-------------------------------------|
| MLP                  | 11.2889 | 16.5765 | 0.859542 | 140.223 | 109.867 | tabular neural-network baseline      |
| STGCN                | 11.2189 | 16.6052 | 0.859614 | 136.059 | 109.083 | spatio-temporal graph neural network |
| Random Forest        | 11.3012 | 17.3312 | 0.846461 | 120.12  | 110.13  | nonlinear tree baseline              |
| HistGradientBoosting | 12.1442 | 18.1475 | 0.831658 | 129.85  | 116.112 | strong structured-data baseline      |
| Ridge                | 17.2014 | 31.0619 | 0.506808 | 176.868 | 123.626 | linear baseline                      |

STGCN 并非本次 RMSE 最低模型，但经过 OD 混合图、增强节点特征和残差目标训练后，相比初始 STGCN 已明显改善。本轮 RMSE 最低模型是 MLP。STGCN 未取得第一的可能原因包括：区域划分较粗、Top 区域数量有限、净流量噪声较大、图结构仍是简化的距离与 OD 混合图，以及当前 STGCN 架构较轻量。

![模型指标对比图](figures/regional_model_metrics_bar.png)

![STGCN 真实值与预测值对比](figures/stgcn_actual_vs_predicted.png)

![STGCN 训练损失曲线](figures/stgcn_training_loss.png)

## 8. 拥堵与调度风险分析

基于 STGCN 的 `predicted_net_flow` 判断调度风险。阈值默认使用训练集中 `abs(net_flow_next_hour)` 的 75% 分位数，也可在配置文件中手动设置。

- `predicted_net_flow < -threshold`：`shortage_risk`，建议提前补车。
- `predicted_net_flow > threshold`：`overflow_risk`，表示还车压力偏高，建议提前移走车辆或预留空桩，缓解满桩拥堵。
- 其他情况：`balanced`，暂不需要明显调度。

调度优先级：

`dispatch_priority = abs(predicted_net_flow) * historical_avg_demand`

满桩拥堵优先级：

`congestion_score = max(predicted_net_flow, 0) * historical_avg_demand * (1 + transit_congestion_index / 10)`

### Top 10 缺车风险区域

| datetime            | grid_id      |   grid_center_lat |   grid_center_lng |   predicted_net_flow | risk_type     |   dispatch_priority |   congestion_score |   nearest_subway_distance |   subway_count_500m |   subway_count_1000m | suggested_action   |
|:--------------------|:-------------|------------------:|------------------:|---------------------:|:--------------|--------------------:|-------------------:|--------------------------:|--------------------:|---------------------:|:-------------------|
| 2026-04-14 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -600.013 | shortage_risk |            109404   |                  0 |                   152.005 |                   6 |                   21 | 建议提前补车             |
| 2026-05-26 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -584.848 | shortage_risk |            106639   |                  0 |                   152.005 |                   6 |                   21 | 建议提前补车             |
| 2026-05-19 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -566.042 | shortage_risk |            103210   |                  0 |                   152.005 |                   6 |                   21 | 建议提前补车             |
| 2026-05-26 17:00:00 | 40.74_-73.99 |             40.74 |            -73.99 |             -350.703 | shortage_risk |             99772.1 |                  0 |                   150.379 |                   3 |                   16 | 建议提前补车             |
| 2026-05-27 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -541.458 | shortage_risk |             98727.5 |                  0 |                   152.005 |                   6 |                   21 | 建议提前补车             |
| 2026-05-05 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -540.837 | shortage_risk |             98614.4 |                  0 |                   152.005 |                   6 |                   21 | 建议提前补车             |
| 2026-05-20 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -533.095 | shortage_risk |             97202.8 |                  0 |                   152.005 |                   6 |                   21 | 建议提前补车             |
| 2026-05-12 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -528.936 | shortage_risk |             96444.4 |                  0 |                   152.005 |                   6 |                   21 | 建议提前补车             |
| 2026-05-18 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -528.537 | shortage_risk |             96371.6 |                  0 |                   152.005 |                   6 |                   21 | 建议提前补车             |
| 2026-05-04 17:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |             -513.966 | shortage_risk |             93714.8 |                  0 |                   152.005 |                   6 |                   21 | 建议提前补车             |

### Top 10 满桩/拥堵风险区域

| datetime            | grid_id      |   grid_center_lat |   grid_center_lng |   predicted_net_flow | risk_type     |   dispatch_priority |   congestion_score |   nearest_subway_distance |   subway_count_500m |   subway_count_1000m | suggested_action     |
|:--------------------|:-------------|------------------:|------------------:|---------------------:|:--------------|--------------------:|-------------------:|--------------------------:|--------------------:|---------------------:|:---------------------|
| 2026-04-15 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              512.402 | overflow_risk |             93429.6 |             337082 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，缓解满桩拥堵 |
| 2026-05-19 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              502.484 | overflow_risk |             91621.1 |             330557 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，缓解满桩拥堵 |
| 2026-05-20 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              496.221 | overflow_risk |             90479.3 |             326438 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，缓解满桩拥堵 |
| 2026-04-14 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              493.285 | overflow_risk |             89943.9 |             324506 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，缓解满桩拥堵 |
| 2026-04-14 08:00:00 | 40.74_-73.99 |             40.74 |            -73.99 |              316.002 | overflow_risk |             89900   |             262058 |                   150.379 |                   3 |                   16 | 建议提前移走车辆或预留空桩，缓解满桩拥堵 |
| 2026-05-06 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              492.328 | overflow_risk |             89769.4 |             323877 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，缓解满桩拥堵 |
| 2026-04-28 08:00:00 | 40.74_-73.99 |             40.74 |            -73.99 |              306.929 | overflow_risk |             87318.8 |             254533 |                   150.379 |                   3 |                   16 | 建议提前移走车辆或预留空桩，缓解满桩拥堵 |
| 2026-05-28 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              473.437 | overflow_risk |             86324.8 |             311449 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，缓解满桩拥堵 |
| 2026-05-05 08:00:00 | 40.74_-73.99 |             40.74 |            -73.99 |              303.198 | overflow_risk |             86257.2 |             251439 |                   150.379 |                   3 |                   16 | 建议提前移走车辆或预留空桩，缓解满桩拥堵 |
| 2026-04-29 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              471.924 | overflow_risk |             86049   |             310454 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，缓解满桩拥堵 |

### Top 10 地铁相关拥堵优先区域

| datetime            | grid_id      |   grid_center_lat |   grid_center_lng |   predicted_net_flow | risk_type                |   dispatch_priority |   congestion_score |   nearest_subway_distance |   subway_count_500m |   subway_count_1000m | suggested_action            |
|:--------------------|:-------------|------------------:|------------------:|---------------------:|:-------------------------|--------------------:|-------------------:|--------------------------:|--------------------:|---------------------:|:----------------------------|
| 2026-04-15 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              512.402 | overflow_congestion_risk |             93429.6 |             337082 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，优先保障地铁/商业周边区域 |
| 2026-05-19 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              502.484 | overflow_congestion_risk |             91621.1 |             330557 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，优先保障地铁/商业周边区域 |
| 2026-05-20 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              496.221 | overflow_congestion_risk |             90479.3 |             326438 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，优先保障地铁/商业周边区域 |
| 2026-04-14 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              493.285 | overflow_congestion_risk |             89943.9 |             324506 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，优先保障地铁/商业周边区域 |
| 2026-05-06 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              492.328 | overflow_congestion_risk |             89769.4 |             323877 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，优先保障地铁/商业周边区域 |
| 2026-05-28 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              473.437 | overflow_congestion_risk |             86324.8 |             311449 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，优先保障地铁/商业周边区域 |
| 2026-04-29 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              471.924 | overflow_congestion_risk |             86049   |             310454 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，优先保障地铁/商业周边区域 |
| 2026-05-21 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              471.292 | overflow_congestion_risk |             85933.8 |             310038 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，优先保障地铁/商业周边区域 |
| 2026-05-05 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              468.097 | overflow_congestion_risk |             85351.1 |             307936 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，优先保障地铁/商业周边区域 |
| 2026-05-12 08:00:00 | 40.76_-73.98 |             40.76 |            -73.98 |              466.169 | overflow_congestion_risk |             84999.6 |             306668 |                   152.005 |                   6 |                   21 | 建议提前移走车辆或预留空桩，优先保障地铁/商业周边区域 |

![STGCN 预测下一小时净流量地图](figures/stgcn_predicted_net_flow_map.png)

![拥堵风险地图](figures/congestion_risk_map.png)

![调度风险地图](figures/dispatch_risk_map.png)

## 9. 可视化分析

![真实净流量地图](figures/regional_net_flow_map.png)

这些图表分别展示区域分布、地铁站空间关系、典型高峰小时取还车差异、真实净流量、STGCN 预测净流量和最终调度风险区域。它们服务于一个结论：区域级净流量和地图特征比城市级订单量更适合指导共享单车再平衡。

## 10. 局限性

1. 当前是区域级预测，不是站点级预测。
2. 没有真实站点容量约束。
3. 没有真实调度车辆路径、人工成本、车辆容量和作业时窗。
4. 邻接矩阵已融合距离和 OD 流量，但仍未使用更复杂的方向性 OD 图、动态 OD 图或站点级流动关系。
5. 地铁站数据质量依赖开放数据字段；如果本地或在线数据不可用，本模块会降级为默认空间特征。
6. GBFS 实时站点状态如果接入，只代表当前状态，不代表历史每小时状态，不能直接当作历史标签。
7. POI/土地利用特征如继续引入，其质量依赖 OSM 数据完整性。

## 11. 改进方向

- 接入 Citi Bike GBFS 实时站点容量和可用车辆数，用于实时风险校正。
- 继续引入 POI、住宅/商业区、办公区等空间特征。
- 使用方向性 OD 流量构造更真实的区域图。
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
- `outputs/tables/region_map_features.csv`
- `outputs/tables/congestion_risk_top10.csv`
- `outputs/models/stgcn_best.pt`
- `outputs/tables/stgcn_predictions.csv`
- `outputs/tables/regional_model_metrics.csv`
- `outputs/tables/dispatch_risk_top10.csv`
- `outputs/tables/spatial_feature_ablation.csv`
- `outputs/tables/spatial_feature_importance.csv`
- `outputs/tables/spatial_feature_analysis.md`
- `outputs/report_stgcn_regional.md`
