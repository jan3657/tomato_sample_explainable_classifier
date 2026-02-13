"""Validation utilities for comparing run output against reference metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReproducibilityValidationReport:
    """Structured reproducibility validation result."""

    passed: bool
    checks: list[dict[str, object]]


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def _close(a: float, b: float, atol: float = 1e-12) -> bool:
    return abs(a - b) <= atol


def validate_reproducibility(baseline_path: str, current_path: str) -> ReproducibilityValidationReport:
    """Validate current run metrics against reference baseline metrics JSON."""
    baseline = _load(baseline_path)
    current = _load(current_path)

    checks: list[dict[str, object]] = []

    checks.append(
        {
            "name": "data_shape",
            "expected": baseline["data"]["shape"],
            "actual": current["data"]["shape"],
            "pass": baseline["data"]["shape"] == current["data"]["shape"],
        }
    )
    checks.append(
        {
            "name": "class_counts",
            "expected": baseline["data"]["class_counts"],
            "actual": current["data"]["class_counts"],
            "pass": baseline["data"]["class_counts"] == current["data"]["class_counts"],
        }
    )

    for key in ("rf_test_mean", "dt_test_mean", "lr_test_mean", "baseline_test_mean"):
        exp = float(baseline["aggregate"][key])
        act = float(current["aggregate"][key])
        checks.append(
            {
                "name": f"aggregate_{key}",
                "expected": exp,
                "actual": act,
                "pass": _close(exp, act),
            }
        )

    for model_key in ("rf_metrics", "dt_metrics", "lr_metrics", "baseline_metrics"):
        exp_rows = baseline[model_key]
        act_rows = current[model_key]
        exact = len(exp_rows) == len(act_rows)
        if exact:
            for exp_row, act_row in zip(exp_rows, act_rows):
                for scalar in ("fold", "test_acc"):
                    if scalar in exp_row and scalar in act_row and exp_row[scalar] != act_row[scalar]:
                        exact = False
                        break
                if "train_acc" in exp_row and "train_acc" in act_row:
                    if not _close(float(exp_row["train_acc"]), float(act_row["train_acc"])):
                        exact = False
                if not exact:
                    break
        checks.append(
            {
                "name": f"{model_key}_rows",
                "expected": exp_rows,
                "actual": act_rows,
                "pass": exact,
            }
        )

    expected_decision_count = len(baseline.get("decision_plot_tasks", []))
    actual_decision_count = int(current.get("decision_plot_count", 0))
    checks.append(
        {
            "name": "decision_plot_count",
            "expected": expected_decision_count,
            "actual": actual_decision_count,
            "pass": expected_decision_count == actual_decision_count,
        }
    )

    expected_tp_count = sum(int(row.get("tp_count", 0)) for row in baseline.get("force_plot_counts", []))
    actual_tp_count = int(current.get("force_tp_plot_count", 0))
    checks.append(
        {
            "name": "force_tp_plot_count",
            "expected": expected_tp_count,
            "actual": actual_tp_count,
            "pass": expected_tp_count == actual_tp_count,
        }
    )

    passed = all(bool(item["pass"]) for item in checks)
    return ReproducibilityValidationReport(passed=passed, checks=checks)


def write_validation_report(report: ReproducibilityValidationReport, output_path: str) -> str:
    """Write validation report as JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "passed": report.passed,
                "checks": report.checks,
            },
            indent=2,
        )
    )
    return str(out)
