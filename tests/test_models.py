"""Unit tests for model construction and training helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tomato_classifier.models import build_models_for_fold, train_and_score


def test_build_models_for_fold_contains_required_models() -> None:
    models = build_models_for_fold(fold_index=2, lr_max_iter=1000, baseline_strategy="most_frequent")
    assert set(models.keys()) == {"rf", "dt", "lr", "baseline"}


def test_train_and_score_returns_valid_accuracy_range() -> None:
    rng = np.random.default_rng(3)
    X_train = pd.DataFrame(rng.normal(size=(30, 6)), columns=[f"f{i}" for i in range(6)])
    X_test = pd.DataFrame(rng.normal(size=(12, 6)), columns=[f"f{i}" for i in range(6)])
    y_train = pd.Series(["A"] * 15 + ["B"] * 15)
    y_test = pd.Series(["A"] * 6 + ["B"] * 6)

    models = build_models_for_fold(fold_index=0, lr_max_iter=1000, baseline_strategy="most_frequent")
    metric = train_and_score(models["dt"], 0, "dt", X_train, y_train, X_test, y_test)

    assert 0.0 <= metric.train_accuracy <= 1.0
    assert 0.0 <= metric.test_accuracy <= 1.0
