"""YAML configuration loader for reproducible run pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .schema import (
    CVConfig,
    ConfusionMatrixVizConfig,
    CorrelationVizConfig,
    DataConfig,
    DecisionPlotVizConfig,
    DendrogramVizConfig,
    FeatureSelectionConfig,
    ForcePlotVizConfig,
    ModelConfig,
    OutputConfig,
    ReproducibilityConfig,
    RocCurveVizConfig,
    RunConfig,
    ShapConfig,
    TreeVizConfig,
    VisualizationConfig,
)


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    """Safely fetch key from mapping with fallback."""
    return d.get(key, default) if d else default


def load_config(path: str) -> RunConfig:
    """Load YAML config into a typed :class:`RunConfig`."""
    raw = yaml.safe_load(Path(path).read_text()) or {}

    data_raw = raw.get("data", {})
    data_cfg = DataConfig(
        path=data_raw["path"],
        sample_col=_get(data_raw, "sample_col", "Sample"),
        target_col=_get(data_raw, "target_col", "Target"),
    )

    cv_raw = raw.get("cv", {})
    cv_cfg = CVConfig(
        n_splits=int(_get(cv_raw, "n_splits", 5)),
        shuffle=bool(_get(cv_raw, "shuffle", True)),
        random_state=int(_get(cv_raw, "random_state", 42)),
    )

    fs_raw = raw.get("feature_selection", {})
    fs_cfg = FeatureSelectionConfig(
        n_attributes=int(_get(fs_raw, "n_attributes", 30)),
        linkage_method=str(_get(fs_raw, "linkage_method", "ward")),
        cluster_criterion=str(_get(fs_raw, "cluster_criterion", "maxclust")),
    )

    model_raw = raw.get("model", {})
    model_cfg = ModelConfig(
        lr_max_iter=int(_get(model_raw, "lr_max_iter", 1000)),
        baseline_strategy=str(_get(model_raw, "baseline_strategy", "most_frequent")),
    )

    shap_raw = raw.get("shap", {})
    shap_cfg = ShapConfig(
        decision_plot_max_samples=_get(shap_raw, "decision_plot_max_samples", None),
        generate_force_tp=bool(_get(shap_raw, "generate_force_tp", True)),
        generate_force_tp_fn_rounded=bool(_get(shap_raw, "generate_force_tp_fn_rounded", True)),
        fail_on_rounded_force_error=bool(_get(shap_raw, "fail_on_rounded_force_error", False)),
    )

    viz_raw = raw.get("visualization", {})
    corr_raw = viz_raw.get("correlation", {})
    den_raw = viz_raw.get("dendrogram", {})
    dec_raw = viz_raw.get("decision_plot", {})
    force_raw = viz_raw.get("force_plot", {})
    tree_raw = viz_raw.get("tree", {})
    cm_raw = viz_raw.get("confusion_matrix", {})
    roc_raw = viz_raw.get("roc_curve", {})
    viz_cfg = VisualizationConfig(
        correlation=CorrelationVizConfig(
            figsize_w=float(_get(corr_raw, "figsize_w", 55.0)),
            figsize_h=float(_get(corr_raw, "figsize_h", 55.0)),
            dpi=int(_get(corr_raw, "dpi", 300)),
        ),
        dendrogram=DendrogramVizConfig(
            figsize_w=float(_get(den_raw, "figsize_w", 30.0)),
            figsize_h=float(_get(den_raw, "figsize_h", 15.0)),
            dpi=int(_get(den_raw, "dpi", 300)),
        ),
        decision_plot=DecisionPlotVizConfig(
            dpi=int(_get(dec_raw, "dpi", 120)),
        ),
        force_plot=ForcePlotVizConfig(
            dpi=int(_get(force_raw, "dpi", 120)),
            rounded_figsize_w=float(_get(force_raw, "rounded_figsize_w", 20.0)),
            rounded_figsize_h=float(_get(force_raw, "rounded_figsize_h", 3.0)),
        ),
        tree=TreeVizConfig(
            figsize_w=float(_get(tree_raw, "figsize_w", 15.0)),
            figsize_h=float(_get(tree_raw, "figsize_h", 15.0)),
            dpi=int(_get(tree_raw, "dpi", 150)),
        ),
        confusion_matrix=ConfusionMatrixVizConfig(
            figsize_w=float(_get(cm_raw, "figsize_w", 10.0)),
            figsize_h=float(_get(cm_raw, "figsize_h", 8.0)),
            dpi=int(_get(cm_raw, "dpi", 200)),
        ),
        roc_curve=RocCurveVizConfig(
            figsize_w=float(_get(roc_raw, "figsize_w", 10.0)),
            figsize_h=float(_get(roc_raw, "figsize_h", 8.0)),
            dpi=int(_get(roc_raw, "dpi", 200)),
        ),
    )

    out_raw = raw.get("output", {})
    out_cfg = OutputConfig(
        root_dir=str(_get(out_raw, "root_dir", "results/reproducible_run")),
        metrics_dir=str(_get(out_raw, "metrics_dir", "metrics")),
        figures_dir=str(_get(out_raw, "figures_dir", "figures")),
        logs_dir=str(_get(out_raw, "logs_dir", "logs")),
    )

    repro_raw = raw.get("reproducibility", {})
    repro_cfg = ReproducibilityConfig(
        global_seed=int(_get(repro_raw, "global_seed", 42)),
    )

    return RunConfig(
        data=data_cfg,
        cv=cv_cfg,
        feature_selection=fs_cfg,
        model=model_cfg,
        shap=shap_cfg,
        visualization=viz_cfg,
        output=out_cfg,
        reproducibility=repro_cfg,
    )
