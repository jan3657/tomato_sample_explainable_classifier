"""Reproducible pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from tomato_classifier.config.schema import RunConfig
from tomato_classifier.data import build_xy, generate_stratified_folds, load_dataset
from tomato_classifier.evaluation import write_json_report, write_markdown_summary
from tomato_classifier.explainability import (
    compute_fold_shap_values,
    generate_all_decision_plots,
    generate_force_plots_rounded_tp_fn,
    generate_force_plots_tp,
)
from tomato_classifier.features import FoldFeatureSelectionResult, select_features_for_fold
from tomato_classifier.models import build_models_for_fold, train_and_score
from tomato_classifier.utils import ensure_output_dirs, get_logger, set_global_seed
from tomato_classifier.visualization import (
    generate_correlation_matrices_by_fold,
    generate_dendrograms_by_fold,
    plot_correlation_matrices_panel,
    plot_confusion_matrix,
    plot_dendrograms_panel,
    plot_last_fold_tree,
    plot_multiclass_roc_curves,
)


@dataclass
class RunArtifacts:
    """Top-level return object for a reproducible run."""

    report_json_path: str
    summary_md_path: str
    payload: dict[str, Any]


def _fold_summary_from_result(result: FoldFeatureSelectionResult) -> dict[str, Any]:
    """Serialize fold feature-selection summary."""
    corr = result.correlation_matrix
    return {
        "fold": result.fold_index,
        "train_size": int(len(result.X_train_selected)),
        "test_size": int(len(result.X_test_selected)),
        "train_class_counts": result.y_train.value_counts().sort_index().to_dict(),
        "test_class_counts": result.y_test.value_counts().sort_index().to_dict(),
        "n_features_before": int(len(result.feature_labels)),
        "n_features_after": int(len(result.representatives)),
        "corr_min": float(np.nanmin(corr.values)),
        "corr_max": float(np.nanmax(corr.values)),
        "cut_distance": result.cut_distance,
        "n_clusters_actual": int(len(np.unique(result.clusters))),
    }


def _metric_dict(metric_obj: Any) -> dict[str, Any]:
    """Normalize metric dataclass to dict schema."""
    return {
        "fold": int(metric_obj.fold_index),
        "train_acc": float(metric_obj.train_accuracy),
        "test_acc": float(metric_obj.test_accuracy),
    }


def run_reproducible_pipeline(config: RunConfig) -> RunArtifacts:
    """Run the full reproducible pipeline and save reports."""
    set_global_seed(config.reproducibility.global_seed)
    dirs = ensure_output_dirs(config.output)
    logger = get_logger("reproducible_run", str(dirs["logs"] / "reproducible_run.log"))

    logger.info("Loading dataset: %s", config.data.path)
    data = load_dataset(config.data.path)
    X, y = build_xy(data, sample_col=config.data.sample_col, target_col=config.data.target_col)

    fold_indices = generate_stratified_folds(
        data,
        target_col=config.data.target_col,
        n_splits=config.cv.n_splits,
        shuffle=config.cv.shuffle,
        random_state=config.cv.random_state,
    )

    fold_results: list[FoldFeatureSelectionResult] = []
    fold_datasets: list[tuple] = []

    logger.info("Running feature selection for %s folds", len(fold_indices))
    for fold in fold_indices:
        train_df = data.iloc[fold.train_index]
        test_df = data.iloc[fold.test_index]

        y_train = train_df[config.data.target_col]
        y_test = test_df[config.data.target_col]
        X_train = train_df.drop(columns=[config.data.sample_col, config.data.target_col])
        X_test = test_df.drop(columns=[config.data.sample_col, config.data.target_col])

        result = select_features_for_fold(
            fold_index=fold.fold_index,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_clusters=config.feature_selection.n_attributes,
            linkage_method=config.feature_selection.linkage_method,
            cluster_criterion=config.feature_selection.cluster_criterion,
        )

        fold_results.append(result)
        fold_datasets.append((result.X_train_selected, result.y_train, result.X_test_selected, result.y_test))

    # Train models and compute SHAP for DT per fold
    rf_metrics: list[dict[str, Any]] = []
    dt_metrics: list[dict[str, Any]] = []
    lr_metrics: list[dict[str, Any]] = []
    baseline_metrics: list[dict[str, Any]] = []

    dt_models: list[object] = []
    fold_shap_values: list[Any] = []
    shap_shapes: list[dict[str, Any]] = []
    model_keys = ("rf", "dt", "lr", "baseline")
    class_labels = sorted([str(label) for label in y.unique()])
    class_to_index = {label: index for index, label in enumerate(class_labels)}
    model_eval_by_fold: dict[str, dict[str, list[np.ndarray]]] = {
        key: {"y_true": [], "y_pred": [], "proba": []} for key in model_keys
    }

    logger.info("Training models and computing SHAP")
    for fold_index, (X_train, y_train, X_test, y_test) in enumerate(fold_datasets):
        models = build_models_for_fold(
            fold_index=fold_index,
            lr_max_iter=config.model.lr_max_iter,
            baseline_strategy=config.model.baseline_strategy,
        )

        rf_metric = train_and_score(models["rf"], fold_index, "rf", X_train, y_train, X_test, y_test)
        dt_metric = train_and_score(models["dt"], fold_index, "dt", X_train, y_train, X_test, y_test)
        lr_metric = train_and_score(models["lr"], fold_index, "lr", X_train, y_train, X_test, y_test)

        models["baseline"].fit(X_train, y_train)
        baseline_test_acc = float((models["baseline"].predict(X_test) == y_test).mean())

        rf_metrics.append(_metric_dict(rf_metric))
        dt_metrics.append(_metric_dict(dt_metric))
        lr_metrics.append(_metric_dict(lr_metric))
        baseline_metrics.append({"fold": fold_index, "test_acc": baseline_test_acc, "majority_class": None})

        y_true_array = y_test.astype(str).to_numpy()
        for model_key in model_keys:
            model = models[model_key]
            y_pred_array = np.asarray(model.predict(X_test)).astype(str)
            model_eval_by_fold[model_key]["y_true"].append(y_true_array)
            model_eval_by_fold[model_key]["y_pred"].append(y_pred_array)

            if hasattr(model, "predict_proba"):
                fold_proba = np.asarray(model.predict_proba(X_test), dtype=float)
                aligned_proba = np.zeros((len(X_test), len(class_labels)), dtype=float)
                for class_index, class_label in enumerate([str(c) for c in model.classes_]):
                    aligned_proba[:, class_to_index[class_label]] = fold_proba[:, class_index]
                model_eval_by_fold[model_key]["proba"].append(aligned_proba)

        dt_model = models["dt"]
        dt_models.append(dt_model)
        shap_result = compute_fold_shap_values(fold_index, dt_model, X_test, y_test)
        fold_shap_values.append(shap_result.shap_values)

        if isinstance(shap_result.shap_values, list):
            shap_shapes.append(
                {
                    "fold": fold_index,
                    "type": "list",
                    "shapes": [list(v.shape) for v in shap_result.shap_values],
                }
            )
        else:
            shap_shapes.append(
                {
                    "fold": fold_index,
                    "type": "array",
                    "shape": list(shap_result.shap_values.shape),
                }
            )

    # Visualization artifacts
    figures_dir = dirs["figures"]
    correlation_matrices = [r.correlation_matrix for r in fold_results]
    linkage_matrices = [r.linkage_matrix for r in fold_results]
    feature_labels_by_fold = [r.feature_labels for r in fold_results]

    correlation_paths = generate_correlation_matrices_by_fold(
        correlation_matrices=correlation_matrices,
        output_dir=str(figures_dir / "correlation_matrices_by_fold"),
        figsize_w=config.visualization.correlation.figsize_w,
        figsize_h=config.visualization.correlation.figsize_h,
        dpi=config.visualization.correlation.dpi,
    )
    correlation_path = plot_correlation_matrices_panel(
        correlation_matrices,
        str(figures_dir / "correlation_matrices_before_reduction.png"),
        figsize_w=config.visualization.correlation.figsize_w,
        figsize_h=config.visualization.correlation.figsize_h,
        dpi=config.visualization.correlation.dpi,
    )

    dendrogram_paths = generate_dendrograms_by_fold(
        linkage_matrices=linkage_matrices,
        feature_labels_per_fold=feature_labels_by_fold,
        correlation_matrices=correlation_matrices,
        n_attributes=config.feature_selection.n_attributes,
        output_dir=str(figures_dir / "dendrograms_by_fold"),
        figsize_w=config.visualization.dendrogram.figsize_w,
        figsize_h=config.visualization.dendrogram.figsize_h,
        dpi=config.visualization.dendrogram.dpi,
    )
    dendrogram_path = plot_dendrograms_panel(
        linkage_matrices,
        feature_labels_by_fold,
        correlation_matrices,
        n_attributes=config.feature_selection.n_attributes,
        out_path=str(figures_dir / "dendrograms_before_reduction.png"),
        figsize_w=config.visualization.dendrogram.figsize_w,
        figsize_h=config.visualization.dendrogram.figsize_h,
        dpi=config.visualization.dendrogram.dpi,
    )
    decision_paths = generate_all_decision_plots(
        fold_datasets=fold_datasets,
        shap_values_by_fold=fold_shap_values,
        dt_models=dt_models,
        output_dir=str(figures_dir / "decision_plots_by_class"),
        dpi=config.visualization.decision_plot.dpi,
        max_samples=config.shap.decision_plot_max_samples,
    )

    force_tp_paths: list[str] = []
    if config.shap.generate_force_tp:
        force_tp_paths = generate_force_plots_tp(
            fold_datasets=fold_datasets,
            dt_models=dt_models,
            shap_values_by_fold=fold_shap_values,
            output_dir=str(figures_dir / "force_plots_tp"),
            dpi=config.visualization.force_plot.dpi,
        )

    rounded_paths: list[str] = []
    rounded_failure: str | None = None
    if config.shap.generate_force_tp_fn_rounded:
        rounded_paths, rounded_failure = generate_force_plots_rounded_tp_fn(
            fold_datasets=fold_datasets,
            dt_models=dt_models,
            shap_values_by_fold=fold_shap_values,
            output_dir=str(figures_dir / "force_plots_rounded"),
            dpi=config.visualization.force_plot.dpi,
            rounded_figsize_w=config.visualization.force_plot.rounded_figsize_w,
            rounded_figsize_h=config.visualization.force_plot.rounded_figsize_h,
            fail_on_error=config.shap.fail_on_rounded_force_error,
        )

    tree_path = plot_last_fold_tree(
        dt_model=dt_models[-1],
        feature_names=list(fold_datasets[-1][2].columns),
        class_names=[str(c) for c in dt_models[-1].classes_],
        out_path=str(figures_dir / "decision_tree_final_fold.png"),
        figsize_w=config.visualization.tree.figsize_w,
        figsize_h=config.visualization.tree.figsize_h,
        dpi=config.visualization.tree.dpi,
    )

    # Count TP/FN per fold/class for artifact traceability.
    force_plot_counts: list[dict[str, Any]] = []
    decision_plot_tasks: list[dict[str, Any]] = []
    for fold_index, (_, _, X_test, y_test) in enumerate(fold_datasets):
        unique_classes = np.unique(y_test)
        y_pred = dt_models[fold_index].predict(X_test)
        for class_index, class_label in enumerate(unique_classes):
            decision_plot_tasks.append(
                {
                    "fold": fold_index,
                    "class_index": class_index,
                    "class_label": str(class_label),
                    "n_samples_class_in_fold_test": int((y_test == class_label).sum()),
                }
            )

            tp_idx = np.where((y_pred == class_label) & (y_test == class_label))[0]
            fn_idx = np.where((y_pred != class_label) & (y_test == class_label))[0]
            force_plot_counts.append(
                {
                    "fold": fold_index,
                    "class_label": str(class_label),
                    "tp_count": int(len(tp_idx)),
                    "fn_count": int(len(fn_idx)),
                }
            )

    mean_test_accuracy = {
        "rf": float(np.mean([row["test_acc"] for row in rf_metrics])),
        "dt": float(np.mean([row["test_acc"] for row in dt_metrics])),
        "lr": float(np.mean([row["test_acc"] for row in lr_metrics])),
        "baseline": float(np.mean([row["test_acc"] for row in baseline_metrics])),
    }
    model_priority = {"dt": 3, "rf": 2, "lr": 1, "baseline": 0}
    best_model_key = max(
        mean_test_accuracy.keys(),
        key=lambda model_key: (mean_test_accuracy[model_key], model_priority.get(model_key, 0)),
    )

    best_y_true = np.concatenate(model_eval_by_fold[best_model_key]["y_true"]).astype(str)
    best_y_pred = np.concatenate(model_eval_by_fold[best_model_key]["y_pred"]).astype(str)
    best_proba: np.ndarray | None = None
    if model_eval_by_fold[best_model_key]["proba"]:
        best_proba = np.vstack(model_eval_by_fold[best_model_key]["proba"])

    best_model_metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(best_y_true, best_y_pred)),
        "precision_macro": float(precision_score(best_y_true, best_y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(best_y_true, best_y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(best_y_true, best_y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(best_y_true, best_y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(best_y_true, best_y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(best_y_true, best_y_pred, average="weighted", zero_division=0)),
    }
    if best_proba is not None:
        try:
            y_true_bin = label_binarize(best_y_true, classes=class_labels)
            if y_true_bin.ndim == 1:
                y_true_bin = y_true_bin.reshape(-1, 1)
            if len(class_labels) == 2 and y_true_bin.shape[1] == 1:
                y_true_bin = np.column_stack((1 - y_true_bin[:, 0], y_true_bin[:, 0]))

            best_model_metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true_bin, best_proba, average="macro", multi_class="ovr")
            )
            best_model_metrics["roc_auc_ovr_weighted"] = float(
                roc_auc_score(y_true_bin, best_proba, average="weighted", multi_class="ovr")
            )
            best_model_metrics["roc_auc_micro"] = float(roc_auc_score(y_true_bin.ravel(), best_proba.ravel()))
        except ValueError as exc:
            logger.warning("Skipping ROC-AUC scalar metrics for %s: %s", best_model_key, exc)

    confusion_matrix_path = plot_confusion_matrix(
        y_true=best_y_true,
        y_pred=best_y_pred,
        class_labels=class_labels,
        out_path=str(figures_dir / f"confusion_matrix_{best_model_key}.png"),
        figsize_w=config.visualization.confusion_matrix.figsize_w,
        figsize_h=config.visualization.confusion_matrix.figsize_h,
        dpi=config.visualization.confusion_matrix.dpi,
    )

    roc_curve_path: str | None = None
    if best_proba is not None:
        try:
            roc_curve_path = plot_multiclass_roc_curves(
                y_true=best_y_true,
                y_score=best_proba,
                class_labels=class_labels,
                out_path=str(figures_dir / f"roc_curves_{best_model_key}.png"),
                figsize_w=config.visualization.roc_curve.figsize_w,
                figsize_h=config.visualization.roc_curve.figsize_h,
                dpi=config.visualization.roc_curve.dpi,
            )
        except ValueError as exc:
            logger.warning("Skipping ROC curve generation for %s: %s", best_model_key, exc)

    payload: dict[str, Any] = {
        "data": {
            "path": config.data.path,
            "shape": [int(data.shape[0]), int(data.shape[1])],
            "n_features_raw": int(X.shape[1]),
            "class_counts": y.value_counts().sort_index().to_dict(),
            "classes_sorted": sorted([str(c) for c in y.unique()]),
        },
        "params": {
            "num_selected_features": config.feature_selection.n_attributes,
            "num_splits": config.cv.n_splits,
            "splitter": (
                f"StratifiedKFold(n_splits={config.cv.n_splits}, "
                f"shuffle={config.cv.shuffle}, random_state={config.cv.random_state})"
            ),
            "rf": "RandomForestClassifier(random_state=fold_index, all other params default)",
            "dt": "DecisionTreeClassifier(random_state=fold_index, all other params default)",
            "lr": (
                f"Pipeline(StandardScaler + LogisticRegression(max_iter={config.model.lr_max_iter}, "
                "random_state=fold_index))"
            ),
            "dummy": f'DummyClassifier(strategy="{config.model.baseline_strategy}")',
        },
        "fold_summaries": [_fold_summary_from_result(result) for result in fold_results],
        "rf_metrics": rf_metrics,
        "dt_metrics": dt_metrics,
        "lr_metrics": lr_metrics,
        "baseline_metrics": baseline_metrics,
        "shap_shapes": shap_shapes,
        "decision_plot_tasks": decision_plot_tasks,
        "force_plot_counts": force_plot_counts,
        "aggregate": {
            "rf_test_mean": mean_test_accuracy["rf"],
            "dt_test_mean": mean_test_accuracy["dt"],
            "lr_test_mean": mean_test_accuracy["lr"],
            "baseline_test_mean": mean_test_accuracy["baseline"],
        },
        "best_model": {
            "name": best_model_key,
            "mean_test_accuracy": mean_test_accuracy[best_model_key],
            "metrics": best_model_metrics,
        },
        "best_model_metrics": best_model_metrics,
        "artifacts": {
            "correlation_panel": correlation_path,
            "correlation_paths": correlation_paths,
            "dendrogram_panel": dendrogram_path,
            "dendrogram_paths": dendrogram_paths,
            "decision_plot_paths": decision_paths,
            "force_tp_paths": force_tp_paths,
            "force_rounded_paths": rounded_paths,
            "tree_path": tree_path,
            "confusion_matrix_path": confusion_matrix_path,
            "roc_curve_path": roc_curve_path,
            "rounded_force_failure": rounded_failure,
        },
        "dataset_path": config.data.path,
        "n_samples": int(data.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "mean_test_accuracy": mean_test_accuracy,
        "best_model_name": best_model_key,
        "correlation_plot_count": len(correlation_paths),
        "dendrogram_plot_count": len(dendrogram_paths),
        "decision_plot_count": len(decision_paths),
        "force_tp_plot_count": len(force_tp_paths),
        "force_rounded_plot_count": len(rounded_paths),
        "confusion_matrix_plot_count": 1 if confusion_matrix_path else 0,
        "roc_curve_plot_count": 1 if roc_curve_path else 0,
        "rounded_force_failure": rounded_failure,
    }

    json_path = write_json_report(payload, str(dirs["metrics"] / "run_metrics.json"))
    summary_path = write_markdown_summary(payload, str(dirs["metrics"] / "run_summary.md"))

    logger.info("Reproducible run complete. Report: %s", json_path)
    return RunArtifacts(report_json_path=json_path, summary_md_path=summary_path, payload=payload)
