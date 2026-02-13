"""Metric aggregation and reporting helpers."""

from .metrics import aggregate_accuracy_by_model, fold_metric_rows
from .reports import write_json_report, write_markdown_summary

__all__ = ["fold_metric_rows", "aggregate_accuracy_by_model", "write_json_report", "write_markdown_summary"]
