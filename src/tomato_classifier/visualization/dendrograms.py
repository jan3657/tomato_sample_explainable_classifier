"""Feature dendrogram panel plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster


def _representatives_for_dendrogram(
    clusters: np.ndarray,
    feature_labels: list[str],
    corr: pd.DataFrame,
) -> set[str]:
    reps: set[str] = set()
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) == 1:
            reps.add(feature_labels[cluster_indices[0]])
            continue

        cluster_cols = [feature_labels[idx] for idx in cluster_indices]
        best_col = max(cluster_cols, key=lambda c: float(corr.loc[c, cluster_cols].sum()))
        reps.add(best_col)
    return reps


def _axes_list(axes_obj: object) -> list[object]:
    """Normalize matplotlib axes container to a list."""
    if isinstance(axes_obj, (list, tuple)):
        return list(axes_obj)
    if hasattr(axes_obj, "ravel"):
        return list(axes_obj.ravel())
    return [axes_obj]


def _plot_single_dendrogram(
    linkage_matrix: np.ndarray,
    feature_labels: list[str],
    corr: pd.DataFrame,
    n_attributes: int,
    ax: object,
    fold_index: int,
    leaf_font_size: int,
    title_font_size: int,
) -> None:
    """Render a fold-specific dendrogram onto an axis."""
    clusters = fcluster(linkage_matrix, t=n_attributes, criterion="maxclust")
    representatives = _representatives_for_dendrogram(clusters, feature_labels, corr)

    row_for_target = len(feature_labels) - n_attributes
    if 0 <= row_for_target < len(linkage_matrix):
        cut_distance = linkage_matrix[row_for_target, 2]
    else:
        cut_distance = None

    dendrogram(
        linkage_matrix,
        labels=feature_labels,
        leaf_rotation=90,
        leaf_font_size=leaf_font_size,
        orientation="top",
        p=n_attributes,
        color_threshold=cut_distance,
        ax=ax,
    )
    if cut_distance is not None:
        ax.axhline(y=cut_distance, c="r", lw=1, ls="-")

    for tick_label in ax.get_xticklabels():
        if tick_label.get_text() in representatives:
            tick_label.set_color("red")

    ax.tick_params(axis="x", labelsize=leaf_font_size)
    ax.tick_params(axis="y", labelsize=max(leaf_font_size - 1, 6))
    ax.set_title(f"Fold {fold_index + 1}", fontsize=title_font_size)


def plot_dendrogram_for_fold(
    linkage_matrix: np.ndarray,
    feature_labels: list[str],
    corr: pd.DataFrame,
    n_attributes: int,
    fold_index: int,
    out_path: str,
    figsize_w: float,
    figsize_h: float,
    dpi: int,
) -> str:
    """Plot and save one fold-specific feature dendrogram."""
    single_figsize_w = max(10.0, min(figsize_w, figsize_h * 2.5))
    single_figsize_h = max(8.0, figsize_h * 1.6)
    fig, ax = plt.subplots(1, 1, figsize=(single_figsize_w, single_figsize_h), dpi=dpi)
    _plot_single_dendrogram(
        linkage_matrix=linkage_matrix,
        feature_labels=feature_labels,
        corr=corr,
        n_attributes=n_attributes,
        ax=ax,
        fold_index=fold_index,
        leaf_font_size=9,
        title_font_size=14,
    )
    plt.tight_layout()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def generate_dendrograms_by_fold(
    linkage_matrices: list[np.ndarray],
    feature_labels_per_fold: list[list[str]],
    correlation_matrices: list[pd.DataFrame],
    n_attributes: int,
    output_dir: str,
    figsize_w: float,
    figsize_h: float,
    dpi: int,
) -> list[str]:
    """Generate one feature dendrogram per fold."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for fold_index, linkage_matrix in enumerate(linkage_matrices):
        out_path = out_dir / f"fold_{fold_index}_dendrogram.png"
        saved_paths.append(
            plot_dendrogram_for_fold(
                linkage_matrix=linkage_matrix,
                feature_labels=feature_labels_per_fold[fold_index],
                corr=correlation_matrices[fold_index],
                n_attributes=n_attributes,
                fold_index=fold_index,
                out_path=str(out_path),
                figsize_w=figsize_w,
                figsize_h=figsize_h,
                dpi=dpi,
            )
        )
    return saved_paths


def plot_dendrograms_panel(
    linkage_matrices: list[np.ndarray],
    feature_labels_per_fold: list[list[str]],
    correlation_matrices: list[pd.DataFrame],
    n_attributes: int,
    out_path: str,
    figsize_w: float,
    figsize_h: float,
    dpi: int,
) -> str:
    """Plot a multi-panel feature dendrogram by fold."""
    panel_figsize_h = max(figsize_h, 6.0)
    fig, axes = plt.subplots(1, len(linkage_matrices), figsize=(figsize_w, panel_figsize_h), dpi=dpi)
    fig.suptitle("Feature Dendrograms for Each Fold (Before Reduction)", fontsize=16)

    axes_list = _axes_list(axes)
    for idx, linkage_matrix in enumerate(linkage_matrices):
        _plot_single_dendrogram(
            linkage_matrix=linkage_matrix,
            feature_labels=feature_labels_per_fold[idx],
            corr=correlation_matrices[idx],
            n_attributes=n_attributes,
            ax=axes_list[idx],
            fold_index=idx,
            leaf_font_size=7,
            title_font_size=11,
        )

    plt.tight_layout()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output)
