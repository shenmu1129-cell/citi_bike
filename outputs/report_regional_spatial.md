# Citi Bike 区域级地图与空间特征调度风险分析

## 摘要

本项目将原先的城市级订单预测改造为区域级共享单车调度风险分析。城市级订单量只能告诉我们全市下一小时需求大概是多少，却无法回答“哪些区域会缺车、哪些区域会满桩”。调度问题的核心是空间不平衡，因此本次使用 Citi Bike 起终点经纬度划分网格区域，并预测每个区域下一小时净流量。

## 数据与样本

| 指标                  | 数值                                        |
|:--------------------|:------------------------------------------|
| 建模样本数               | 613847                                    |
| 区域数量                | 299                                       |
| 时间范围                | 2026-03-07 10:00:00 至 2026-05-31 22:00:00 |
| MTA 地铁站数量           | 485                                       |
| GBFS 站点数量           | 2411                                      |
| OSM/Overpass POI 数量 | 38130                                     |

## 为什么城市级预测不能直接指导调度

城市级订单预测会把曼哈顿、布鲁克林、皇后区等区域的流入流出抵消掉。例如 A 区大量取车、B 区大量还车时，全市订单量可能稳定，但 A 区会缺车、B 区会满桩。调度车辆需要知道空间位置、风险方向和优先级，而不是单一总量。

## 为什么引入空间特征

共享单车需求受周边功能强烈影响。地铁站附近通勤换乘明显，餐饮和商业 POI 影响午间与晚间活动，学校、公园和住宅区对应不同出行节奏，实时站点容量和空桩数则直接影响下一小时的服务风险。因此区域级模型同时使用历史净流量、时间特征、地图 POI、地铁站距离和 GBFS 实时站点状态。

## 地图与空间数据源

- Citi Bike tripdata：使用 `start_lat`、`start_lng`、`end_lat`、`end_lng` 构造区域级 pickup、dropoff 和 net flow。
- MTA Subway Stations 开放数据：计算最近地铁站距离、500 米和 1000 米地铁站数量。
- OpenStreetMap / Overpass API：统计每个区域 500 米范围内 shop、restaurant、cafe、school、park、residential、commercial 数量，并缓存 API 结果。
- Citi Bike GBFS：读取 station_information 和 station_status，统计每个区域站点数、总容量、当前可用车辆、空桩和电单车数量。

## 区域净流量定义

经纬度按 `0.01` 度固定网格聚合，每个 `grid_id` 表示一个区域：

`net_flow = dropoff_count - pickup_count`

主预测目标为：

`net_flow_next_hour = net_flow.groupby(grid_id).shift(-1)`

当预测净流量为负，说明下一小时该区域取车多于还车，可能缺车；当预测净流量为正，说明还车多于取车，可能满桩。

## 模型结果

按时间顺序切分训练集和测试集，未使用随机切分。对比 Linear Regression、Random Forest、HistGradientBoosting 和 MLP，主模型采用 HistGradientBoosting。本次最佳 RMSE 模型为 **MLP**，RMSE=7.759，R2=0.7816。

| model                |     MAE |     RMSE |       R2 |     MAPE |   sMAPE |
|:---------------------|--------:|---------:|---------:|---------:|--------:|
| MLP                  | 4.05008 |  7.75948 | 0.781563 |  96.2516 | 116.901 |
| Random Forest        | 3.99973 |  8.07527 | 0.763421 |  88.0054 | 109.044 |
| HistGradientBoosting | 4.25327 |  8.41148 | 0.743311 |  91.6045 | 117.056 |
| Linear Regression    | 5.23389 | 12.446   | 0.438017 | 124.561  | 124.273 |

![模型指标对比图](figures/regional_model_metrics_bar.png)

## 调度风险规则

阈值使用训练集中 `abs(net_flow_next_hour)` 的 75% 分位数。若 `predicted_net_flow_next_hour < -threshold`，标记为 `shortage_risk`；若 `predicted_net_flow_next_hour > threshold`，标记为 `overflow_risk`。

`dispatch_priority = abs(predicted_net_flow_next_hour) * historical_avg_demand`

该优先级同时考虑风险强度和历史需求规模。

## Top 10 缺车风险区域

| datetime_hour       | grid_id      |   grid_lat |   grid_lng | risk_type     |   predicted_net_flow_next_hour |   actual_net_flow_next_hour |   historical_avg_demand |   dispatch_priority |
|:--------------------|:-------------|-----------:|-----------:|:--------------|-------------------------------:|----------------------------:|------------------------:|--------------------:|
| 2026-05-31 22:00:00 | 40.74_-73.99 |      40.74 |     -73.99 | shortage_risk |                      -22.5418  |                         -19 |                 289.125 |            6517.39  |
| 2026-05-31 22:00:00 | 40.72_-73.96 |      40.72 |     -73.96 | shortage_risk |                      -23.646   |                         -21 |                 185.177 |            4378.69  |
| 2026-05-31 22:00:00 | 40.73_-73.99 |      40.73 |     -73.99 | shortage_risk |                      -11.3177  |                          12 |                 327.63  |            3708.03  |
| 2026-05-31 22:00:00 | 40.73_-74.00 |      40.73 |     -74    | shortage_risk |                      -12.9573  |                         -22 |                 226.524 |            2935.14  |
| 2026-05-31 22:00:00 | 40.72_-74.00 |      40.72 |     -74    | shortage_risk |                       -9.01021 |                           1 |                 181.357 |            1634.07  |
| 2026-05-31 22:00:00 | 40.76_-73.98 |      40.76 |     -73.98 | shortage_risk |                       -7.65409 |                           7 |                 185.832 |            1422.37  |
| 2026-05-31 22:00:00 | 40.75_-73.99 |      40.75 |     -73.99 | shortage_risk |                       -4.8273  |                           8 |                 270.563 |            1306.09  |
| 2026-05-31 22:00:00 | 40.69_-73.99 |      40.69 |     -73.99 | shortage_risk |                       -6.7686  |                          15 |                 124.051 |             839.654 |
| 2026-05-31 22:00:00 | 40.77_-73.98 |      40.77 |     -73.98 | shortage_risk |                       -4.43344 |                           3 |                 188.202 |             834.384 |
| 2026-05-31 22:00:00 | 40.75_-73.98 |      40.75 |     -73.98 | shortage_risk |                       -4.34256 |                           6 |                 189.194 |             821.585 |

## Top 10 满桩风险区域

| datetime_hour       | grid_id      |   grid_lat |   grid_lng | risk_type     |   predicted_net_flow_next_hour |   actual_net_flow_next_hour |   historical_avg_demand |   dispatch_priority |
|:--------------------|:-------------|-----------:|-----------:|:--------------|-------------------------------:|----------------------------:|------------------------:|--------------------:|
| 2026-05-31 22:00:00 | 40.72_-73.98 |      40.72 |     -73.98 | overflow_risk |                       23.0562  |                          31 |                158.094  |            3645.04  |
| 2026-05-31 22:00:00 | 40.73_-73.98 |      40.73 |     -73.98 | overflow_risk |                       13.1255  |                          30 |                231.213  |            3034.78  |
| 2026-05-31 22:00:00 | 40.76_-73.99 |      40.76 |     -73.99 | overflow_risk |                        6.35901 |                          59 |                213.017  |            1354.58  |
| 2026-05-31 22:00:00 | 40.74_-73.98 |      40.74 |     -73.98 | overflow_risk |                        4.76976 |                          30 |                225.693  |            1076.5   |
| 2026-05-31 22:00:00 | 40.77_-73.99 |      40.77 |     -73.99 | overflow_risk |                        7.38887 |                          22 |                114.44   |             845.58  |
| 2026-05-31 22:00:00 | 40.77_-73.95 |      40.77 |     -73.95 | overflow_risk |                        7.75011 |                          10 |                 74.4468 |             576.971 |
| 2026-05-31 22:00:00 | 40.69_-73.97 |      40.69 |     -73.97 | overflow_risk |                        6.10471 |                          -4 |                 85.6063 |             522.602 |
| 2026-05-31 22:00:00 | 40.69_-73.96 |      40.69 |     -73.96 | overflow_risk |                        8.46698 |                          24 |                 54.3495 |             460.176 |
| 2026-05-31 22:00:00 | 40.70_-73.93 |      40.7  |     -73.93 | overflow_risk |                       10.7355  |                          12 |                 40.9712 |             439.844 |
| 2026-05-31 22:00:00 | 40.68_-73.96 |      40.68 |     -73.96 | overflow_risk |                        5.11398 |                           6 |                 67.8302 |             346.882 |

## 可视化

![区域网格地图](figures/regional_grid_map.png)

![地铁站与 Citi Bike 区域叠加图](figures/subway_bike_grid_map.png)

![POI 密度地图](figures/poi_density_map.png)

![预测净流量地图](figures/predicted_net_flow_map.png)

## 补充数据图、实验图与对比图

![区域聚合后的每小时取车与还车趋势](figures/regional_hourly_pickup_dropoff_trend.png)

![区域小时净流量分布](figures/regional_net_flow_distribution.png)

![星期-小时平均绝对净流量热力图](figures/regional_weekday_hour_abs_net_flow_heatmap.png)

![模型实验对比：RMSE 与 R2](figures/regional_model_rmse_r2_comparison.png)

![模型绝对误差分布对比](figures/regional_model_absolute_error_boxplot.png)

![HistGradientBoosting 真实值 vs 预测值](figures/regional_hgb_actual_vs_predicted_scatter.png)

![Top 调度风险区域优先级对比](figures/regional_top_dispatch_priority_bar.png)

![空间特征与区域平均绝对净流量的相关性](figures/regional_spatial_feature_correlation_bar.png)

## 局限性

1. GBFS 是实时状态，只代表脚本运行时的站点状态，不等同于 2026 年历史每小时真实库存。
2. POI 数据质量依赖 OpenStreetMap，存在分类不一致和覆盖不完整的问题。
3. 当前模型没有纳入真实调度车辆路径、车辆载重、人工成本和作业时窗。
4. 固定网格比真实运营区简单，边界附近会出现空间归属误差。
5. Overpass 或 GBFS 请求失败时会使用离线降级特征，保证流程运行，但空间解释力会下降。

## 交付文件

- `outputs/tables/regional_spatial_features.csv`
- `outputs/tables/regional_model_metrics.csv`
- `outputs/tables/dispatch_risk_top10.csv`
- `outputs/figures/regional_grid_map.png`
- `outputs/figures/subway_bike_grid_map.png`
- `outputs/figures/poi_density_map.png`
- `outputs/figures/predicted_net_flow_map.png`
- `outputs/report_regional_spatial.md`
