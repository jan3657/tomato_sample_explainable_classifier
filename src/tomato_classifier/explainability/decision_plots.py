"""SHAP decision plot generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .shap_utils import get_class_expected_value, get_class_shap_values


def plot_decision_plot_for_all_samples(
    fold_index: int,
    class_index: int,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    shap_values_fold: Any,
    dt_model: object,
    output_dir: Path,
    dpi: int,
    max_samples: int | None = None,
) -> Path:
    """Generate one decision plot for one fold/class pair."""
    unique_classes = np.unique(y_test)
    class_label = unique_classes[class_index]

    X_plot = X_test
    shap_values_local = shap_values_fold
    if max_samples is not None and max_samples < len(X_test):
        sample_indices = np.random.choice(len(X_test), size=max_samples, replace=False)
        X_plot = X_test.iloc[sample_indices]
        if isinstance(shap_values_local, list):
            shap_values_local = [values[sample_indices] for values in shap_values_local]
        else:
            shap_values_local = shap_values_local[sample_indices]

    explainer = shap.TreeExplainer(dt_model)
    class_shap_values = get_class_shap_values(shap_values_local, class_index)
    expected_value = get_class_expected_value(explainer.expected_value, class_index)

    plt.figure(figsize=(12, 8))
    shap.decision_plot(
        expected_value,
        class_shap_values,
        X_plot,
        feature_names=X_plot.columns.tolist(),
        show=False,
    )
    plt.title(f"SHAP Decision Plot for All Samples in Fold {fold_index}, Class {class_label}", fontsize=14)
    plt.tight_layout()

    class_dir = output_dir / f"class_{class_label}"
    class_dir.mkdir(parents=True, exist_ok=True)
    out_path = class_dir / f"fold_{fold_index}_class_{class_label}.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return out_path


def generate_all_decision_plots(
    fold_datasets: list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]],
    shap_values_by_fold: list[Any],
    dt_models: list[object],
    output_dir: str,
    dpi: int,
    max_samples: int | None,
) -> list[str]:
    """Generate decision plots for all folds and all classes."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for fold_index, (_, _, X_test, y_test) in enumerate(fold_datasets):
        unique_classes = np.unique(y_test)
        for class_index, _ in enumerate(unique_classes):
            path = plot_decision_plot_for_all_samples(
                fold_index=fold_index,
                class_index=class_index,
                X_test=X_test,
                y_test=y_test,
                shap_values_fold=shap_values_by_fold[fold_index],
                dt_model=dt_models[fold_index],
                output_dir=out_dir,
                dpi=dpi,
                max_samples=max_samples,
            )
            saved_paths.append(str(path))
    return saved_paths
