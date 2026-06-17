# Server Training Guide

This repository keeps the Citi Bike code and configuration only. Raw Citi Bike zip files, processed CSV files, model weights, and generated figures are intentionally ignored by git.

## Setup

```bash
git clone https://github.com/shenmu1129-cell/citi_bike.git
cd citi_bike
conda create -n wwt310 python=3.10 -y
conda run -n wwt310 python -m pip install -r requirements.txt
```

## Run Original City-Level Pipeline

```bash
conda run -n wwt310 python run_all.py
```

## Run Regional STGCN Pipeline

Full 12-month run:

```bash
conda run -n wwt310 python run_stgcn_regional.py
```

Quick smoke test:

```bash
conda run -n wwt310 python run_stgcn_regional.py --quick
```

If raw zip files are already present in `data/raw/citibike/`:

```bash
conda run -n wwt310 python run_stgcn_regional.py --skip-download
```

Regenerate figures and report from existing outputs:

```bash
conda run -n wwt310 python run_stgcn_regional.py --report-only
```

## Main Outputs

- `data/processed/regional_hourly_panel.csv`
- `data/processed/regional_model_dataset.csv`
- `outputs/tables/regional_model_metrics.csv`
- `outputs/tables/stgcn_predictions.csv`
- `outputs/tables/dispatch_risk_top10.csv`
- `outputs/models/stgcn_best.pt`
- `outputs/report_stgcn_regional.md`
