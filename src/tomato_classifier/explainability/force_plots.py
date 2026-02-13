"""SHAP force plot generation for TP and FN cases."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .shap_utils import get_class_expected_value, get_class_shap_values


def _force_plot_with_runtime_warning_filter(
    *,
    base_value: float,
    shap_values: np.ndarray,
    features: pd.Series,
    feature_names: list[str],
    figsize: tuple[float, float] | None = None,
) -> None:
    """Render SHAP force plot while suppressing known zero-effect runtime warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in scalar divide",
            category=RuntimeWarning,
        )
        kwargs: dict[str, Any] = {
            "base_value": base_value,
            "shap_values": shap_values,
            "features": features,
            "feature_names": feature_names,
            "matplotlib": True,
            "show": False,
        }
        if figsize is not None:
            kwargs["figsize"] = figsize
        shap.force_plot(**kwargs)


def generate_force_plots_tp(
    fold_datasets: list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]],
    dt_models: list[object],
    shap_values_by_fold: list[Any],
    output_dir: str,
    dpi: int,
) -> list[str]:
    """Generate true-positive SHAP force plots with figure titles."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    for fold_index, (_, _, X_test, y_test) in enumerate(fold_datasets):
        dt_model = dt_models[fold_index]
        y_pred = dt_model.predict(X_test)
        unique_classes = np.unique(y_test)
        explainer = shap.TreeExplainer(dt_model)

        for class_index, class_label in enumerate(unique_classes):
            tp_indices = np.where((y_pred == class_label) & (y_test == class_label))[0]
            if len(tp_indices) == 0:
                continue

            expected_value_for_class = get_class_expected_value(explainer.expected_value, class_index)
            class_shap_values = get_class_shap_values(shap_values_by_fold[fold_index], class_index)[tp_indices, :]
            X_test_tp = X_test.iloc[tp_indices]

            for i, sample_index in enumerate(tp_indices):
                single_sample = X_test_tp.iloc[i, :]
                single_shap = class_shap_values[i, :]
                fig = None
                try:
                    _force_plot_with_runtime_warning_filter(
                        base_value=expected_value_for_class,
                        shap_values=single_shap,
                        features=single_sample,
                        feature_names=single_sample.index.tolist(),
                    )
                    fig = plt.gcf()
                    plt.title(f"Fold {fold_index + 1} - Class {class_label} - TP Sample {sample_index}", fontsize=12)
                    plt.tight_layout()
                    out_path = out_dir / f"fold_{fold_index}_class_{class_label}_TP_{sample_index}.png"
                    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                    saved_paths.append(str(out_path))
                finally:
                    if fig is not None:
                        plt.close(fig)
                    else:
                        plt.close()

    return saved_paths


def generate_force_plots_rounded_tp_fn(
    fold_datasets: list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]],
    dt_models: list[object],
    shap_values_by_fold: list[Any],
    output_dir: str,
    dpi: int,
    rounded_figsize_w: float,
    rounded_figsize_h: float,
    fail_on_error: bool,
) -> tuple[list[str], str | None]:
    """Generate rounded TP/FN force plots with optional fail-fast behavior."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    failure_message: str | None = None

    for fold_index, (_, _, X_test, y_test) in enumerate(fold_datasets):
        dt_model = dt_models[fold_index]
        y_pred = dt_model.predict(X_test)
        unique_classes = np.unique(y_test)
        explainer = shap.TreeExplainer(dt_model)

        for class_index, class_label in enumerate(unique_classes):
            expected_value_for_class = get_class_expected_value(explainer.expected_value, class_index)
            class_shap_all = get_class_shap_values(shap_values_by_fold[fold_index], class_index)

            tp_indices = np.where((y_pred == class_label) & (y_test == class_label))[0]
            fn_indices = np.where((y_pred != class_label) & (y_test == class_label))[0]

            for sample_index in tp_indices:
                single_sample = X_test.iloc[sample_index, :]
                single_shap = class_shap_all[sample_index, :]
                sample_display = single_sample.round(3)

                fig = None
                try:
                    _force_plot_with_runtime_warning_filter(
                        base_value=expected_value_for_class,
                        shap_values=single_shap,
                        features=sample_display,
                        feature_names=single_sample.index.tolist(),
                        figsize=(rounded_figsize_w, rounded_figsize_h),
                    )
                    fig = plt.gcf()
                    plt.title(
                        f"Fold {fold_index + 1} - Class {class_label} - TP Sample {sample_index}",
                        fontsize=14,
                        y=1.5,
                    )
                    out_path = out_dir / f"fold_{fold_index}_class_{class_label}_TP_{sample_index}.png"
                    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                    saved_paths.append(str(out_path))
                finally:
                    if fig is not None:
                        plt.close(fig)
                    else:
                        plt.close()

            for sample_index in fn_indices:
                single_sample = X_test.iloc[sample_index, :]
                single_shap = class_shap_all[sample_index, :]
                sample_display = single_sample.round(3)
                predicted_as = y_pred[sample_index]

                fig = None
                try:
                    _force_plot_with_runtime_warning_filter(
                        base_value=expected_value_for_class,
                        shap_values=single_shap,
                        features=sample_display,
                        feature_names=single_sample.index.tolist(),
                        figsize=(rounded_figsize_w, rounded_figsize_h),
                    )
                    fig = plt.gcf()
                    plt.title(
                        (
                            f"Fold {fold_index + 1} - Class {class_label} - "
                            f"FN Sample {sample_index} (predicted as {predicted_as})"
                        ),
                        fontsize=14,
                        y=1.5,
                    )
                    out_path = out_dir / f"fold_{fold_index}_class_{class_label}_FN_{sample_index}.png"
                    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                    saved_paths.append(str(out_path))
                except Exception as exc:  # noqa: BLE001 - plotting backend can fail per sample
                    failure_message = str(exc)
                    if fail_on_error:
                        return saved_paths, failure_message
                finally:
                    if fig is not None:
                        plt.close(fig)
                    else:
                        plt.close()

    return saved_paths, failure_message
