"""Cross-validation split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class FoldIndices:
    """Train/test index arrays for one fold."""

    fold_index: int
    train_index: np.ndarray
    test_index: np.ndarray


def generate_stratified_folds(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int,
    shuffle: bool,
    random_state: int,
) -> list[FoldIndices]:
    """Generate deterministic stratified folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds: list[FoldIndices] = []
    for fold_index, (train_idx, test_idx) in enumerate(skf.split(df, df[target_col])):
        folds.append(FoldIndices(fold_index, train_idx, test_idx))
    return folds
