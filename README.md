# Citi Bike 小时需求预测与区域级 STGCN 调度风险扩展

固定位置：

`/Users/shenmu/Downloads/同步空间/Mac/CodexWorkspace/citibike_demand_prediction_rerun`

## 原城市级需求预测入口

原有城市级小时需求预测、Open-Meteo 天气融合、传统模型对比和 `outputs/report.md` 保留不变。

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

## 新增区域级 STGCN 调度风险扩展

新增 `run_stgcn_regional.py`，在原项目基础上增量构造区域级小时面板、区域 kNN 图、传统模型对比和 STGCN 主模型，用于预测每个区域下一小时净流量，并识别缺车/满桩风险。

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
- `outputs/report_stgcn_regional.md`

## 输出

- `data/raw/citibike/`：Citi Bike 原始 zip
- `data/raw/weather/weather_raw.json`：Open-Meteo 天气原始数据
- `data/processed/model_dataset.csv`：最终建模数据
- `outputs/figures/`：EDA 与模型评估图
- `outputs/tables/model_metrics.csv`：模型指标
- `outputs/report.md`：课程报告 Markdown
