# 基于 Citi Bike 历史骑行数据与天气 API 的城市共享单车小时需求预测研究

## 摘要

本项目围绕城市共享单车小时需求预测问题，重新构建了完整的数据科学流程。项目使用纽约 Citi Bike 官方历史骑行数据，并结合 Open-Meteo Historical Weather API 的小时级天气数据，完成数据采集、清洗、小时聚合、天气融合、探索性分析、特征工程、模型训练、模型评估与报告输出。

研究任务定义为监督学习回归问题：给定当前小时的时间特征、天气特征和历史需求特征，预测下一小时全市 Citi Bike 骑行订单量。实验比较 Linear Regression、Random Forest、HistGradientBoosting 和 MLP 模型。本次重跑结果中，按 RMSE 排名的最佳模型为 **MLP**，RMSE=765.514，R2=0.944315。

## 1. 数据来源

- Citi Bike 官方 tripdata：`202603`、`202604`、`202605`
- Open-Meteo Historical Weather API：纽约经纬度 40.7128, -74.0060
- 数据粒度：城市级小时需求
- 建模目标：`target_next_hour = rental_count.shift(-1)`

## 2. EDA 摘要

| 指标      | 数值                                        |
|:--------|:------------------------------------------|
| 样本时间范围  | 2026-03-07 10:00:00 到 2026-05-31 22:00:00 |
| 总小时数    | 2053                                      |
| 总骑行量    | 6944189                                   |
| 平均小时骑行量 | 3382.46                                   |
| 最大小时骑行量 | 12759                                     |
| 缺失值数量   | 0                                         |
| 下雨小时占比  | 0.1364                                    |
| 周末小时占比  | 0.2986                                    |

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

| model                |     MAE |    RMSE |       R2 |    MAPE |   sMAPE |
|:---------------------|--------:|--------:|---------:|--------:|--------:|
| MLP                  | 555.181 | 765.514 | 0.944315 | 32.5857 | 24.6274 |
| Random Forest        | 490.397 | 772.799 | 0.94325  | 16.3182 | 15.0857 |
| HistGradientBoosting | 539.178 | 821.747 | 0.935834 | 16.9933 | 15.5745 |
| Linear Regression    | 632.13  | 846.602 | 0.931893 | 40.003  | 28.4542 |

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
