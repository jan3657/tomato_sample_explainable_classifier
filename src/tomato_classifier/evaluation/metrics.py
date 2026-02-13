"""Accuracy metric aggregation utilities."""

from __future__ import annotations

from typing import Iterable


def fold_metric_rows(metric_records: Iterable[dict]) -> list[dict]:
    """Normalize metric records as serializable row dicts."""
    rows: list[dict] = []
    for rec in metric_records:
        rows.append(
            {
                "fold": int(rec["fold"]),
                "model": str(rec["model"]),
                "train_accuracy": float(rec["train_accuracy"]),
                "test_accuracy": float(rec["test_accuracy"]),
            }
        )
    return rows


def aggregate_accuracy_by_model(metric_rows: Iterable[dict]) -> dict[str, float]:
    """Return mean test accuracy by model."""
    grouped: dict[str, list[float]] = {}
    for row in metric_rows:
        grouped.setdefault(row["model"], []).append(float(row["test_accuracy"]))
    return {model: sum(values) / len(values) for model, values in grouped.items() if values}
