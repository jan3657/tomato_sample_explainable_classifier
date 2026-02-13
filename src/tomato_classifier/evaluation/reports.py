"""Report writers for reproducible run artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def write_json_report(payload: dict, output_path: str) -> str:
    """Write JSON payload to file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    return str(out)


def write_markdown_summary(payload: dict, output_path: str) -> str:
    """Write a concise markdown summary of run metrics."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Reproducible Run Summary",
        "",
        f"- Dataset: `{payload['dataset_path']}`",
        f"- Samples: {payload['n_samples']}",
        f"- Raw feature count: {payload['n_features_raw']}",
        f"- Best model: `{payload.get('best_model_name', 'n/a')}`",
        "",
        "## Mean Test Accuracy",
    ]
    for model_name, value in payload.get("mean_test_accuracy", {}).items():
        lines.append(f"- {model_name}: {value:.6f}")

    best_metrics = payload.get("best_model_metrics", {})
    if best_metrics:
        lines.extend(
            [
                "",
                "## Best Model Metrics",
                f"- Accuracy: {best_metrics.get('accuracy', 0.0):.6f}",
                f"- Precision (macro): {best_metrics.get('precision_macro', 0.0):.6f}",
                f"- Precision (weighted): {best_metrics.get('precision_weighted', 0.0):.6f}",
                f"- Recall (macro): {best_metrics.get('recall_macro', 0.0):.6f}",
                f"- Recall (weighted): {best_metrics.get('recall_weighted', 0.0):.6f}",
                f"- F1 (macro): {best_metrics.get('f1_macro', 0.0):.6f}",
                f"- F1 (weighted): {best_metrics.get('f1_weighted', 0.0):.6f}",
            ]
        )
        if "roc_auc_ovr_macro" in best_metrics:
            lines.append(f"- ROC-AUC OVR (macro): {best_metrics['roc_auc_ovr_macro']:.6f}")
        if "roc_auc_ovr_weighted" in best_metrics:
            lines.append(f"- ROC-AUC OVR (weighted): {best_metrics['roc_auc_ovr_weighted']:.6f}")
        if "roc_auc_micro" in best_metrics:
            lines.append(f"- ROC-AUC (micro): {best_metrics['roc_auc_micro']:.6f}")

    lines.extend(
        [
            "",
            "## Plot Outputs",
            f"- Correlation plots (per-fold): {payload.get('correlation_plot_count', 0)}",
            f"- Dendrogram plots (per-fold): {payload.get('dendrogram_plot_count', 0)}",
            f"- Decision plots: {payload.get('decision_plot_count', 0)}",
            f"- Force plots (TP): {payload.get('force_tp_plot_count', 0)}",
            f"- Force plots (rounded TP/FN): {payload.get('force_rounded_plot_count', 0)}",
            f"- Confusion matrix plots: {payload.get('confusion_matrix_plot_count', 0)}",
            f"- ROC curve plots: {payload.get('roc_curve_plot_count', 0)}",
            "",
            f"Rounded-force failure: `{payload.get('rounded_force_failure')}`",
        ]
    )

    out.write_text("\n".join(lines) + "\n")
    return str(out)
