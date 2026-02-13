"""SHAP computation and class-specific extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap


@dataclass
class ShapFoldResult:
    """SHAP artifacts for one fold."""

    fold_index: int
    shap_values: Any
    expected_value: Any
    classes: np.ndarray


def compute_fold_shap_values(fold_index: int, dt_model: object, X_test: pd.DataFrame, y_test: pd.Series) -> ShapFoldResult:
    """Compute SHAP values with TreeExplainer for one fold."""
    explainer = shap.TreeExplainer(dt_model)
    shap_values = explainer.shap_values(X_test)
    classes = np.unique(y_test)
    return ShapFoldResult(
        fold_index=fold_index,
        shap_values=shap_values,
        expected_value=explainer.expected_value,
        classes=classes,
    )


def get_class_shap_values(shap_values: Any, class_index: int) -> np.ndarray:
    """Return class-specific SHAP matrix across SHAP return formats."""
    if isinstance(shap_values, list):
        return shap_values[class_index]
    return shap_values[:, :, class_index]


def get_class_expected_value(expected_value: Any, class_index: int) -> float:
    """Return scalar expected value for one class."""
    if isinstance(expected_value, (list, np.ndarray)):
        return float(expected_value[class_index])
    return float(expected_value)
