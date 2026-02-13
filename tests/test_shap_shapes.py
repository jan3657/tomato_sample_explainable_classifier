"""Unit tests for SHAP shape handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from tomato_classifier.explainability.shap_utils import compute_fold_shap_values


def test_compute_fold_shap_values_returns_expected_shape() -> None:
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.normal(size=(40, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(["A"] * 20 + ["B"] * 20)

    model = DecisionTreeClassifier(random_state=0)
    model.fit(X, y)

    result = compute_fold_shap_values(0, model, X.iloc[:8], y.iloc[:8])
    sv = result.shap_values

    if isinstance(sv, list):
        assert len(sv) >= 2
        assert sv[0].shape[0] == 8
        assert sv[0].shape[1] == 5
    else:
        assert sv.shape[0] == 8
        assert sv.shape[1] == 5
