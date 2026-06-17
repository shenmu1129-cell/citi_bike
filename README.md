# Citi Bike 区域拥堵与调度风险分析

固定位置：

`/Users/shenmu/Downloads/同步空间/Mac/CodexWorkspace/citibike_demand_prediction_rerun`

## 原城市级需求预测入口

原有城市级小时需求预测代码保留不变，便于对照历史版本。但当前项目汇报主线已调整为区域级拥堵/缺车风险和共享单车调度分析；天气特征不再作为新增区域模型的核心输入。

```bash
python3 run_all.py
```

快速冒烟测试：

```bash
python3 run_all.py --quick
```

如果原始 zip 已经存在，只想重新处理和建模：

```bash
python3 run_all.py --skip-download
```

## 区域级 STGCN 拥堵与调度风险模块

新增 `run_stgcn_regional.py`，在原项目基础上增量构造区域级小时面板、地图空间特征、距离 + OD 混合图、传统模型对比和完整 STGCN 主模型。目标是预测每个区域下一小时净流量，并识别缺车风险与满桩拥堵风险。

核心特征包括：

- Citi Bike 起终点经纬度划分的 `grid_id`
- 区域每小时 `pickup_count`、`dropoff_count`、`net_flow`
- 会员/临时用户、车型结构、rush hour、lag、rolling 特征
- MTA 地铁站地图特征：最近地铁距离、500m/1000m 地铁站数量、地铁相关拥堵指数
- 区域距离 kNN + 历史 OD 流量混合图结构

本机执行请使用 `wwt310` conda 环境：

```bash
conda run -n wwt310 python run_stgcn_regional.py --skip-download
```

快速冒烟测试：

```bash
conda run -n wwt310 python run_stgcn_regional.py --quick --skip-download
```

其他运行方式：

```bash
conda run -n wwt310 python run_stgcn_regional.py --train-only
conda run -n wwt310 python run_stgcn_regional.py --report-only
```

主要新增输出：

- `data/processed/regional_hourly_panel.csv`
- `data/processed/regional_model_dataset.csv`
- `outputs/tables/region_grid_info.csv`
- `outputs/tables/region_adjacency_matrix.csv`
- `outputs/tables/region_edges.csv`
- `outputs/models/stgcn_best.pt`
- `outputs/tables/stgcn_predictions.csv`
- `outputs/tables/regional_model_metrics.csv`
- `outputs/tables/dispatch_risk_top10.csv`
- `outputs/tables/congestion_risk_top10.csv`
- `outputs/tables/region_map_features.csv`
- `outputs/report_stgcn_regional.md`

## 输出

- `data/raw/citibike/`：Citi Bike 原始 zip
- `data/raw/spatial/`：MTA/GBFS 等空间数据缓存
- `data/processed/model_dataset.csv`：最终建模数据
- `outputs/figures/`：EDA 与模型评估图
- `outputs/tables/model_metrics.csv`：模型指标
- `outputs/report.md`：课程报告 Markdown
