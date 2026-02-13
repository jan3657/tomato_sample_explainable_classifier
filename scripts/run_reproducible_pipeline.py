#!/usr/bin/env python3
"""Run the reproducible pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tomato_classifier.config import load_config
from tomato_classifier.pipeline import run_reproducible_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible pipeline")
    parser.add_argument("--config", default="configs/reproducible_run.yaml", help="YAML config path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts = run_reproducible_pipeline(cfg)
    print(f"Run report: {artifacts.report_json_path}")
    print(f"Summary: {artifacts.summary_md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
