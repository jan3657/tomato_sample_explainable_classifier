"""Feature-selection artifacts tracked per fold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FoldFeatureSelectionResult:
    """Per-fold output from clustering-based selection."""

    fold_index: int
    X_train_selected: pd.DataFrame
    y_train: pd.Series
    X_test_selected: pd.DataFrame
    y_test: pd.Series
    correlation_matrix: pd.DataFrame
    linkage_matrix: np.ndarray
    feature_labels: list[str]
    clusters: np.ndarray
    representatives: list[str]
    cut_distance: float | None
