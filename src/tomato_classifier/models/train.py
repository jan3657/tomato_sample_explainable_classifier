"""Per-fold model training and accuracy metrics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import accuracy_score


@dataclass
class ModelFoldMetrics:
    """Train/test accuracy for one model and fold."""

    fold_index: int
    model_name: str
    train_accuracy: float
    test_accuracy: float


def train_and_score(
    model: object,
    fold_index: int,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelFoldMetrics:
    """Fit model and return fold-level accuracy metrics."""
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return ModelFoldMetrics(
        fold_index=fold_index,
        model_name=model_name,
        train_accuracy=float(accuracy_score(y_train, y_pred_train)),
        test_accuracy=float(accuracy_score(y_test, y_pred_test)),
    )
