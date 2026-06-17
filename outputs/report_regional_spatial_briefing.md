# Citi Bike 区域级调度风险分析汇报稿

## 1. 汇报主线

本次项目不再做“全市下一小时有多少订单”的城市级预测，而是转向更贴近运营的问题：**下一小时哪些区域可能缺车，哪些区域可能满桩，以及应该优先调度哪些区域**。

核心思路是把 Citi Bike 骑行记录中的起终点经纬度映射到固定网格区域，计算每个区域每小时的取车数、还车数和净流量，再结合地铁站、POI、GBFS 站点状态等空间特征，预测区域下一小时净流量，并给出缺车/满桩风险。

## 2. 建议 PPT 结构

### 第 1 页：题目与问题定义

题目：Citi Bike 区域级地图与空间特征调度风险分析

要讲清楚：

- 旧问题是城市级订单预测。
- 新问题是区域级调度风险判断。
- 调度真正关心的不是总订单量，而是空间不平衡。

口播示例：

> 之前的模型能预测全市下一小时骑行需求，但这个结果不能直接告诉运营人员该往哪里调车。共享单车调度的核心问题是区域之间的供需错配，所以本次项目把目标改成预测每个区域下一小时净流量，并识别缺车和满桩风险。

建议图：

![区域网格地图](figures/regional_grid_map.png)

### 第 2 页：数据图：为什么城市级预测不够

要讲清楚：

- 城市级订单会抵消区域差异。
- A 区大量取车、B 区大量还车时，全市总量可能正常。
- 但运营上 A 区缺车、B 区满桩都需要处理。

建议图：

![区域聚合后的每小时取车与还车趋势](figures/regional_hourly_pickup_dropoff_trend.png)

口播示例：

> 城市级预测只给一个总量，无法区分需求发生在哪里。调度车辆需要知道具体区域的位置、风险方向和优先级。因此我们把预测粒度从城市级下沉到网格区域级。

### 第 3 页：数据图：区域划分与目标变量

要讲清楚：

- 使用 `start_lat`、`start_lng`、`end_lat`、`end_lng`。
- 经纬度按 `0.01` 度固定网格聚合。
- 每个 `grid_id` 是一个区域。
- 每小时每区域计算：
  - `pickup_count`
  - `dropoff_count`
  - `net_flow = dropoff_count - pickup_count`
- 预测目标是 `net_flow_next_hour`。

关键解释：

- `net_flow < 0`：取车多于还车，区域车辆减少，可能缺车。
- `net_flow > 0`：还车多于取车，区域车辆增加，可能满桩。

建议图：

![区域小时净流量分布](figures/regional_net_flow_distribution.png)

![星期-小时平均绝对净流量热力图](figures/regional_weekday_hour_abs_net_flow_heatmap.png)

口播示例：

> 我们把每条骑行记录拆成起点取车和终点还车两个事件，然后按小时和网格区域聚合。净流量定义为还车数减取车数。这个指标比订单数更适合调度，因为它直接描述区域车辆库存的变化方向。

### 第 4 页：数据图：空间特征设计

要讲清楚三类空间特征分别有什么用：

MTA 地铁站特征：

- `nearest_subway_distance`
- `subway_count_500m`
- `subway_count_1000m`
- 解释通勤换乘需求。

OSM / Overpass POI 特征：

- shop、restaurant、cafe、school、park、residential、commercial
- 解释区域功能属性和活动密度。

GBFS 实时站点状态：

- `station_count`
- `total_capacity`
- `current_bikes`
- `current_empty_docks`
- `current_ebikes`
- 辅助判断当前服务能力和风险承载能力。

建议图：

![地铁站与 Citi Bike 区域叠加图](figures/subway_bike_grid_map.png)

![POI 密度地图](figures/poi_density_map.png)

![空间特征与区域平均绝对净流量的相关性](figures/regional_spatial_feature_correlation_bar.png)

口播示例：

> 地铁站影响通勤换乘，POI 反映区域功能，GBFS 则提供当前站点容量和车辆状态。这些变量让模型不只是看历史流量，还能理解一个区域为什么会产生某种供需变化。

### 第 5 页：数据规模与建模流程

关键数字：

- 建模样本数：`613,847`
- 区域数量：`299`
- 时间范围：`2026-03-07 10:00:00` 至 `2026-05-31 22:00:00`
- MTA 地铁站：`485`
- GBFS 站点：`2,411`
- OSM/Overpass POI：`38,130`

建模流程：

1. 读取 Citi Bike 原始 tripdata。
2. 网格化起终点经纬度。
3. 聚合区域小时级取车、还车和净流量。
4. 加入地铁、POI、GBFS 空间特征。
5. 按 `grid_id` 分组构造下一小时目标，避免区域错位。
6. 按时间顺序切分训练集和测试集。
7. 对比 Linear Regression、Random Forest、HistGradientBoosting、MLP。

口播示例：

> 这里特别注意两点：第一，目标变量按 grid_id 分组 shift，避免不同区域之间错位；第二，训练和测试按时间顺序切分，避免随机切分造成未来信息泄漏。

### 第 6 页：模型结果

模型指标：

| 模型 | MAE | RMSE | R2 | MAPE | sMAPE |
|---|---:|---:|---:|---:|---:|
| MLP | 4.050 | 7.759 | 0.782 | 96.252 | 116.901 |
| Random Forest | 4.000 | 8.075 | 0.763 | 88.005 | 109.044 |
| HistGradientBoosting | 4.253 | 8.411 | 0.743 | 91.605 | 117.056 |
| Linear Regression | 5.234 | 12.446 | 0.438 | 124.561 | 124.273 |

讲法建议：

- MLP 的 RMSE 最低。
- Random Forest MAE 略低，整体也较强。
- HistGradientBoosting 作为主模型，兼顾非线性能力、稳定性和解释性。
- Linear Regression 明显较弱，说明区域净流量存在非线性关系。

建议图：

![模型实验对比：RMSE 与 R2](figures/regional_model_rmse_r2_comparison.png)

![模型指标对比图](figures/regional_model_metrics_bar.png)

![模型绝对误差分布对比](figures/regional_model_absolute_error_boxplot.png)

![HistGradientBoosting 真实值 vs 预测值](figures/regional_hgb_actual_vs_predicted_scatter.png)

这页至少放 `regional_model_rmse_r2_comparison.png` 和 `regional_hgb_actual_vs_predicted_scatter.png`，能同时展示“模型对比”和“主模型预测效果”。

口播示例：

> 从结果看，非线性模型明显优于线性回归，说明区域净流量受到时间、历史流量、空间属性和站点状态的共同影响。虽然 MLP 的 RMSE 最低，但项目中仍使用 HistGradientBoosting 作为主调度模型，因为它更稳定，也更适合后续做特征解释和规则化部署。

### 第 7 页：调度风险规则与结果

风险规则：

- 阈值使用训练集中 `abs(net_flow_next_hour)` 的 75% 分位数。
- `predicted_net_flow_next_hour < -threshold`：`shortage_risk`，可能缺车。
- `predicted_net_flow_next_hour > threshold`：`overflow_risk`，可能满桩。
- `dispatch_priority = abs(predicted_net_flow_next_hour) * historical_avg_demand`

Top 缺车风险区域：

- `40.74_-73.99`
- `40.72_-73.96`
- `40.73_-73.99`
- `40.73_-74.00`
- `40.72_-74.00`

Top 满桩风险区域：

- `40.72_-73.98`
- `40.73_-73.98`
- `40.76_-73.99`
- `40.74_-73.98`
- `40.77_-73.99`

建议图：

![预测净流量地图](figures/predicted_net_flow_map.png)

![Top 调度风险区域优先级对比](figures/regional_top_dispatch_priority_bar.png)

表格可引用：`outputs/tables/dispatch_risk_top10.csv`

口播示例：

> 预测值为负代表下一小时车辆会净流出，因此可能缺车；预测值为正代表车辆会净流入，因此可能满桩。为了排序，我们不仅看预测净流量绝对值，还乘以该区域历史平均需求，这样可以优先处理风险强、需求也高的区域。

### 第 8 页：结论与局限性

结论：

- 区域级净流量比城市级订单量更适合调度分析。
- 空间特征能帮助模型理解区域功能和供需结构。
- 模型可以输出可执行的 Top 风险区域，而不是只输出总量预测。
- 预测地图和风险表可以作为调度决策的前置输入。

局限性：

- GBFS 是实时状态，不代表历史每小时真实库存。
- POI 数据质量依赖 OSM。
- 没有真实调度车辆路径、人工成本和车辆载重约束。
- 固定网格不是正式运营边界，边缘区域可能存在归属误差。

口播示例：

> 这个项目完成了从需求预测到调度风险识别的转变。它不能直接替代调度优化系统，但可以为调度系统提供区域级风险输入。下一步可以把预测结果接入车辆路径规划，进一步考虑车辆容量、人工成本和调度时窗。

## 3. 图表清单

### 数据图

- `outputs/figures/regional_grid_map.png`：区域网格地图，说明研究粒度从城市级下沉到区域级。
- `outputs/figures/regional_hourly_pickup_dropoff_trend.png`：每小时取车和还车趋势，说明数据规模和时间变化。
- `outputs/figures/regional_net_flow_distribution.png`：区域小时净流量分布，说明预测目标的取值形态。
- `outputs/figures/regional_weekday_hour_abs_net_flow_heatmap.png`：星期-小时绝对净流量热力图，说明调度压力具有时间模式。
- `outputs/figures/subway_bike_grid_map.png`：地铁站与单车区域叠加，说明地铁空间特征。
- `outputs/figures/poi_density_map.png`：POI 密度地图，说明区域功能差异。

### 实验图

- `outputs/figures/regional_model_rmse_r2_comparison.png`：模型 RMSE 与 R2 对比，适合放在实验结果页。
- `outputs/figures/regional_model_metrics_bar.png`：MAE、RMSE、sMAPE 多指标对比。
- `outputs/figures/regional_hgb_actual_vs_predicted_scatter.png`：主模型真实值与预测值散点图。
- `outputs/figures/regional_model_absolute_error_boxplot.png`：四个模型绝对误差分布对比。

### 调度结果图

- `outputs/figures/predicted_net_flow_map.png`：预测净流量空间分布图。
- `outputs/figures/regional_top_dispatch_priority_bar.png`：Top 缺车/满桩风险区域优先级条形图。
- `outputs/figures/regional_spatial_feature_correlation_bar.png`：空间特征与区域净流量强度的相关性对比。

## 4. 1 分钟极简版总结

本项目从城市级订单预测转向区域级调度风险分析。我们使用 Citi Bike 原始数据中的起终点经纬度，把纽约划分为 0.01 度网格区域，并按小时统计每个区域的取车数、还车数和净流量。净流量定义为还车数减取车数，负值表示车辆减少、可能缺车，正值表示车辆增加、可能满桩。

为了提高预测能力，我们加入了三类空间特征：MTA 地铁站距离和数量，用于刻画通勤换乘；OpenStreetMap POI，用于刻画商业、餐饮、学校、公园、住宅等区域功能；Citi Bike GBFS 实时站点状态，用于表示当前站点容量、可用车和空桩情况。

模型方面，我们对比了 Linear Regression、Random Forest、HistGradientBoosting 和 MLP，并按时间顺序划分训练集和测试集，避免未来信息泄漏。结果显示非线性模型明显优于线性模型，其中 MLP 的 RMSE 最低，HistGradientBoosting 作为主调度模型也取得了较稳定的效果。

最后，我们根据预测的下一小时净流量设置缺车和满桩风险规则，并用 `dispatch_priority = abs(predicted_net_flow_next_hour) * historical_avg_demand` 对区域排序，输出 Top 10 缺车风险区域和 Top 10 满桩风险区域。这样模型结果就从单纯预测数字，转化成了可以辅助调度决策的区域清单和风险地图。

## 5. 答辩可能被问到的问题

Q：为什么不用城市级订单预测做调度？

A：城市级订单预测只有总量，没有空间方向。调度需要知道具体哪里缺车、哪里满桩。区域之间的流入流出会在全市总量里互相抵消，所以城市级预测不能直接指导调度。

Q：为什么用净流量作为目标？

A：调度关心的是车辆库存变化。取车会减少区域车辆，还车会增加区域车辆，所以 `dropoff_count - pickup_count` 能直接反映区域库存压力。

Q：为什么按 `grid_id` 分组 shift？

A：如果直接全表 shift，上一行和下一行可能属于不同区域，目标就会错位。按 `grid_id` 分组后，`net_flow_next_hour` 才表示同一个区域下一小时的净流量。

Q：GBFS 是实时数据，会不会有问题？

A：这是一个局限。GBFS 反映脚本运行时的实时状态，不代表历史每小时库存。因此它适合作为当前调度风险辅助变量，但不能完全解释历史时段的真实库存变化。

Q：为什么主模型选 HistGradientBoosting，而不是 RMSE 最低的 MLP？

A：MLP 在本次测试中 RMSE 最低，但 HistGradientBoosting 更稳定、训练速度较快，也更容易在表格型特征上部署和解释。因此项目把 HistGradientBoosting 作为主调度模型，同时保留 MLP 作为对比模型。

Q：这个结果能直接派车吗？

A：还不能直接替代调度系统。它提供的是区域级风险输入。真正派车还需要加入车辆路径、车辆容量、司机成本、时间窗和站点级约束。
