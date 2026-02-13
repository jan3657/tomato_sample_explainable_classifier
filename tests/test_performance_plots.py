"""Tests for performance plotting helpers."""

from __future__ import annotations

import numpy as np

from tomato_classifier.visualization.performance import plot_confusion_matrix, plot_multiclass_roc_curves


def test_plot_confusion_matrix_writes_file(tmp_path) -> None:
    y_true = np.array(["A", "B", "A", "B", "A", "B"])
    y_pred = np.array(["A", "A", "A", "B", "B", "B"])
    out_path = tmp_path / "confusion.png"

    saved_path = plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_labels=["A", "B"],
        out_path=str(out_path),
        figsize_w=6.0,
        figsize_h=5.0,
        dpi=100,
    )

    assert out_path.exists()
    assert saved_path == str(out_path)


def test_plot_multiclass_roc_curves_writes_file(tmp_path) -> None:
    y_true = np.array(["A", "B", "C", "A", "B", "C"])
    y_score = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.10, 0.80, 0.10],
            [0.05, 0.20, 0.75],
            [0.70, 0.20, 0.10],
            [0.15, 0.70, 0.15],
            [0.05, 0.25, 0.70],
        ]
    )
    out_path = tmp_path / "roc.png"

    saved_path = plot_multiclass_roc_curves(
        y_true=y_true,
        y_score=y_score,
        class_labels=["A", "B", "C"],
        out_path=str(out_path),
        figsize_w=6.0,
        figsize_h=5.0,
        dpi=100,
    )

    assert out_path.exists()
    assert saved_path == str(out_path)
