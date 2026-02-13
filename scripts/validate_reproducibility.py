#!/usr/bin/env python3
"""Validate reproducibility metrics against reference baseline metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tomato_classifier.pipeline import validate_reproducibility
from tomato_classifier.pipeline.validate import write_validation_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate reproducible output against baseline")
    parser.add_argument(
        "--baseline",
        default="tests/fixtures/reference_metrics.json",
        help="Reference baseline metrics JSON",
    )
    parser.add_argument(
        "--current",
        default="results/reproducible_run/metrics/run_metrics.json",
        help="Current run metrics JSON",
    )
    parser.add_argument(
        "--output",
        default="results/reproducible_run/metrics/reproducibility_validation.json",
        help="Output validation report path",
    )
    args = parser.parse_args()

    report = validate_reproducibility(args.baseline, args.current)
    out_path = write_validation_report(report, args.output)

    print(f"Validation report: {out_path}")
    print(f"Passed: {report.passed}")
    if not report.passed:
        failed = [c for c in report.checks if not c["pass"]]
        print(json.dumps(failed, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
