"""Correlation-matrix panel plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _axes_list(axes_obj: object) -> list[object]:
    """Normalize matplotlib axes container to a list."""
    if isinstance(axes_obj, (list, tuple)):
        return list(axes_obj)
    if hasattr(axes_obj, "ravel"):
        return list(axes_obj.ravel())
    return [axes_obj]


def _set_dense_feature_ticks(ax: object, labels: list[str], fontsize: int) -> None:
    """Show every feature label on both heatmap axes with compact typography."""
    tick_positions = np.arange(len(labels)) + 0.5
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(labels, rotation=0, fontsize=fontsize)


def plot_correlation_matrix_for_fold(
    corr: pd.DataFrame,
    fold_index: int,
    out_path: str,
    figsize_w: float,
    figsize_h: float,
    dpi: int,
) -> str:
    """Plot and save one fold-specific correlation heatmap."""
    plt.figure(figsize=(figsize_w, figsize_h), dpi=dpi)
    ax = sns.heatmap(corr, cmap="coolwarm", square=True, cbar=True)
    _set_dense_feature_ticks(ax=ax, labels=corr.columns.tolist(), fontsize=4)
    plt.title(f"Correlation Matrix (Fold {fold_index + 1})")
    plt.tight_layout()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close()
    return str(output)


def generate_correlation_matrices_by_fold(
    correlation_matrices: list[pd.DataFrame],
    output_dir: str,
    figsize_w: float,
    figsize_h: float,
    dpi: int,
) -> list[str]:
    """Generate one correlation heatmap per fold."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for fold_index, corr in enumerate(correlation_matrices):
        out_path = out_dir / f"fold_{fold_index}_correlation_matrix.png"
        saved_paths.append(
            plot_correlation_matrix_for_fold(
                corr=corr,
                fold_index=fold_index,
                out_path=str(out_path),
                figsize_w=figsize_w,
                figsize_h=figsize_h,
                dpi=dpi,
            )
        )
    return saved_paths


def plot_correlation_matrices_panel(
    correlation_matrices: list[pd.DataFrame],
    out_path: str,
    figsize_w: float,
    figsize_h: float,
    dpi: int,
) -> str:
    """Plot a multi-panel correlation heatmap by fold."""
    fig, axes = plt.subplots(1, len(correlation_matrices), figsize=(figsize_w, figsize_h), dpi=dpi)
    fig.suptitle("Correlation Matrices for Each Fold (Before Reduction)", fontsize=16)

    axes_list = _axes_list(axes)
    for idx, corr in enumerate(correlation_matrices):
        ax = axes_list[idx]
        sns.heatmap(corr, cmap="coolwarm", ax=ax, square=True, cbar=False)
        _set_dense_feature_ticks(ax=ax, labels=corr.columns.tolist(), fontsize=3)
        ax.set_title(f"Fold {idx + 1}")

    plt.tight_layout()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output)
