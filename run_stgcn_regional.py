#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.build_region_graph import build_region_graph
from src.build_regional_panel import build_regional_model_dataset, build_regional_panel, download_months
from src.dispatch_risk import build_dispatch_risk
from src.regional_common import ensure_dirs, load_config, local_citibike_months, log, month_range_ending, resolve_months
from src.report_stgcn_regional import generate_report
from src.train_baselines_regional import train_baselines
from src.train_stgcn import train_stgcn
from src.visualize_regional import make_visualizations


def choose_months(config, quick: bool, skip_download: bool):
    months = resolve_months(config, quick=quick)
    configured = [str(m) for m in config.get("months", []) if str(m).strip()]
    local = local_citibike_months()
    if not quick and not skip_download and not configured:
        latest = local[-1] if local else months[-1]
        months = month_range_ending(latest, 12)
    return months


def run_data_stage(config, quick: bool, skip_download: bool):
    months = choose_months(config, quick, skip_download)
    log(f"regional STGCN months: {months}")
    zip_paths = download_months(months, config, skip_download=skip_download)
    panel = build_regional_panel(zip_paths, config, quick=quick)
    build_regional_model_dataset(panel)
    build_region_graph(config)


def run_train_stage(config, quick: bool):
    train_baselines(config, quick=quick)
    train_stgcn(config, quick=quick)
    build_dispatch_risk(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use recent months and sampled chunks for a fast smoke test.")
    parser.add_argument("--skip-download", action="store_true", help="Use local Citi Bike zip files only.")
    parser.add_argument("--train-only", action="store_true", help="Build data if needed, then train models and risk table only.")
    parser.add_argument("--report-only", action="store_true", help="Regenerate figures and report from existing outputs.")
    parser.add_argument("--config", default="config_stgcn.yaml", help="Path to STGCN config file.")
    args = parser.parse_args()

    ensure_dirs()
    config = load_config(Path(args.config))

    if args.report_only:
        make_visualizations(config)
        generate_report(config)
        log("report-only done")
        return

    run_data_stage(config, quick=args.quick, skip_download=args.skip_download)
    run_train_stage(config, quick=args.quick)

    if not args.train_only:
        make_visualizations(config)
        generate_report(config)

    log("regional STGCN pipeline done")
    log("report: outputs/report_stgcn_regional.md")
    log("model: outputs/models/stgcn_best.pt")
    log("predictions: outputs/tables/stgcn_predictions.csv")


if __name__ == "__main__":
    main()
