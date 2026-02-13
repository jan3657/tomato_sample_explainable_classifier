"""Validation utility tests for reproducibility comparisons."""

from __future__ import annotations

import json

from tomato_classifier.pipeline.validate import validate_reproducibility


def test_validate_reproducibility_with_matching_payload_passes(tmp_path) -> None:
    baseline_path = "tests/fixtures/reference_metrics.json"
    baseline = json.loads(open(baseline_path, "r", encoding="utf-8").read())

    # Build a minimal current payload that matches baseline checks.
    current = {
        "data": baseline["data"],
        "aggregate": baseline["aggregate"],
        "rf_metrics": baseline["rf_metrics"],
        "dt_metrics": baseline["dt_metrics"],
        "lr_metrics": baseline["lr_metrics"],
        "baseline_metrics": baseline["baseline_metrics"],
        "decision_plot_count": len(baseline.get("decision_plot_tasks", [])),
        "force_tp_plot_count": sum(int(row.get("tp_count", 0)) for row in baseline.get("force_plot_counts", [])),
    }

    current_path = tmp_path / "current.json"
    current_path.write_text(json.dumps(current), encoding="utf-8")

    report = validate_reproducibility(baseline_path, str(current_path))
    assert report.passed
