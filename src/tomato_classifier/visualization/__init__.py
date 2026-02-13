"""Plotting utilities for reproducible-run visuals."""

from .correlation import generate_correlation_matrices_by_fold, plot_correlation_matrices_panel
from .dendrograms import generate_dendrograms_by_fold, plot_dendrograms_panel
from .performance import plot_confusion_matrix, plot_multiclass_roc_curves
from .tree import plot_last_fold_tree

__all__ = [
    "plot_correlation_matrices_panel",
    "generate_correlation_matrices_by_fold",
    "plot_dendrograms_panel",
    "generate_dendrograms_by_fold",
    "plot_confusion_matrix",
    "plot_multiclass_roc_curves",
    "plot_last_fold_tree",
]
