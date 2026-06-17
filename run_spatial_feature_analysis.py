#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.analyze_spatial_feature_importance import analyze_spatial_feature_importance
from src.regional_common import ensure_dirs, load_config, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use smaller samples for a fast feature-importance run.")
    parser.add_argument("--config", default="config_stgcn.yaml")
    args = parser.parse_args()

    ensure_dirs()
    config = load_config(Path(args.config))
    analyze_spatial_feature_importance(config, quick=args.quick)
    log("spatial feature analysis done")
    log("table: outputs/tables/spatial_feature_ablation.csv")
    log("table: outputs/tables/spatial_feature_importance.csv")
    log("report: outputs/tables/spatial_feature_analysis.md")


if __name__ == "__main__":
    main()
