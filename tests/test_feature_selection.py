"""Unit tests for reproducible feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tomato_classifier.features import select_features_for_fold


def test_select_features_for_fold_returns_expected_count() -> None:
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.normal(size=(20, 10)), columns=[f"f{i}" for i in range(10)])
    X_test = pd.DataFrame(rng.normal(size=(8, 10)), columns=[f"f{i}" for i in range(10)])
    y_train = pd.Series(["A"] * 10 + ["B"] * 10)
    y_test = pd.Series(["A"] * 4 + ["B"] * 4)

    result = select_features_for_fold(
        fold_index=0,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_clusters=4,
        linkage_method="ward",
        cluster_criterion="maxclust",
    )

    assert len(result.representatives) == 4
    assert result.X_train_selected.shape[1] == 4
    assert result.X_test_selected.shape[1] == 4
    assert result.cut_distance is not None


def test_representatives_are_subset_of_original_columns() -> None:
    rng = np.random.default_rng(7)
    columns = [f"x{i}" for i in range(12)]
    X_train = pd.DataFrame(rng.normal(size=(24, 12)), columns=columns)
    X_test = pd.DataFrame(rng.normal(size=(10, 12)), columns=columns)
    y_train = pd.Series(["C"] * 12 + ["D"] * 12)
    y_test = pd.Series(["C"] * 5 + ["D"] * 5)

    result = select_features_for_fold(
        fold_index=1,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_clusters=6,
        linkage_method="ward",
        cluster_criterion="maxclust",
    )

    assert set(result.representatives).issubset(set(columns))
    assert len(np.unique(result.clusters)) == 6


def test_select_features_for_fold_handles_constant_columns() -> None:
    rng = np.random.default_rng(17)
    columns = [f"x{i}" for i in range(8)]
    X_train = pd.DataFrame(rng.normal(size=(30, 8)), columns=columns)
    X_test = pd.DataFrame(rng.normal(size=(12, 8)), columns=columns)
    # Inject constant features that would otherwise produce NaN correlations.
    X_train["x1"] = 0.0
    X_train["x5"] = 1.0
    X_test["x1"] = 0.0
    X_test["x5"] = 1.0
    y_train = pd.Series(["A"] * 15 + ["B"] * 15)
    y_test = pd.Series(["A"] * 6 + ["B"] * 6)

    result = select_features_for_fold(
        fold_index=0,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_clusters=4,
        linkage_method="ward",
        cluster_criterion="maxclust",
    )

    assert len(result.representatives) == 4
    assert np.isfinite(result.linkage_matrix).all()
