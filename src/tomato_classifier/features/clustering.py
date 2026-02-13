"""Reproducible feature clustering and representative selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .artifacts import FoldFeatureSelectionResult


def compute_abs_correlation(X_train: pd.DataFrame) -> pd.DataFrame:
    """Compute absolute Pearson correlation matrix."""
    feature_labels = X_train.columns.tolist()
    corr = X_train.corr(numeric_only=True).abs()
    corr = corr.reindex(index=feature_labels, columns=feature_labels)
    # Constant features produce NaN correlations; treat them as zero similarity.
    corr = corr.fillna(0.0)
    for feature in feature_labels:
        corr.loc[feature, feature] = 1.0
    return corr


def compute_linkage_from_correlation(corr: pd.DataFrame, method: str = "ward") -> np.ndarray:
    """Build linkage matrix from correlation-derived distance."""
    dist_matrix = 1 - corr
    dist_condensed = squareform(dist_matrix, checks=False)
    return linkage(dist_condensed, method=method)


def select_representatives(
    corr: pd.DataFrame,
    linkage_matrix: np.ndarray,
    feature_labels: list[str],
    n_clusters: int,
    cluster_criterion: str = "maxclust",
) -> tuple[np.ndarray, list[str]]:
    """Select one representative feature per cluster."""
    clusters = fcluster(linkage_matrix, t=n_clusters, criterion=cluster_criterion)
    representatives: list[str] = []

    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        if len(cluster_indices) == 1:
            representatives.append(feature_labels[cluster_indices[0]])
            continue

        cluster_cols = [feature_labels[idx] for idx in cluster_indices]
        best_score = -1.0
        best_col = cluster_cols[0]
        for idx in cluster_indices:
            col_name = feature_labels[idx]
            score = float(corr.loc[col_name, cluster_cols].sum())
            if score > best_score:
                best_score = score
                best_col = col_name
        representatives.append(best_col)

    return clusters, representatives


def _compute_cut_distance(linkage_matrix: np.ndarray, n_features: int, n_clusters: int) -> float | None:
    """Compute dendrogram cut-distance for the requested cluster count."""
    row_for_target = n_features - n_clusters
    if 0 <= row_for_target < len(linkage_matrix):
        return float(linkage_matrix[row_for_target, 2])
    return None


def select_features_for_fold(
    fold_index: int,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_clusters: int,
    linkage_method: str,
    cluster_criterion: str,
) -> FoldFeatureSelectionResult:
    """Execute full per-fold feature-selection workflow."""
    corr = compute_abs_correlation(X_train)
    linkage_matrix = compute_linkage_from_correlation(corr, method=linkage_method)
    feature_labels = X_train.columns.tolist()
    clusters, representatives = select_representatives(
        corr,
        linkage_matrix,
        feature_labels,
        n_clusters=n_clusters,
        cluster_criterion=cluster_criterion,
    )

    X_train_selected = X_train[representatives]
    X_test_selected = X_test[representatives]
    cut_distance = _compute_cut_distance(linkage_matrix, len(feature_labels), n_clusters)

    return FoldFeatureSelectionResult(
        fold_index=fold_index,
        X_train_selected=X_train_selected,
        y_train=y_train,
        X_test_selected=X_test_selected,
        y_test=y_test,
        correlation_matrix=corr,
        linkage_matrix=linkage_matrix,
        feature_labels=feature_labels,
        clusters=clusters,
        representatives=representatives,
        cut_distance=cut_distance,
    )
