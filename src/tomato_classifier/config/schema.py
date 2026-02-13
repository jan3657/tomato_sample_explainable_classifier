"""Typed configuration schema for reproducible experiment runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DataConfig:
    """Dataset location and column names."""

    path: str
    sample_col: str = "Sample"
    target_col: str = "Target"


@dataclass(frozen=True)
class CVConfig:
    """Cross-validation settings."""

    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Feature clustering and representative selection settings."""

    n_attributes: int = 30
    linkage_method: str = "ward"
    cluster_criterion: str = "maxclust"


@dataclass(frozen=True)
class ModelConfig:
    """Model hyperparameters."""

    lr_max_iter: int = 1000
    baseline_strategy: str = "most_frequent"


@dataclass(frozen=True)
class ShapConfig:
    """SHAP plotting and execution settings."""

    decision_plot_max_samples: Optional[int] = None
    generate_force_tp: bool = True
    generate_force_tp_fn_rounded: bool = True
    fail_on_rounded_force_error: bool = False


@dataclass(frozen=True)
class CorrelationVizConfig:
    """Correlation panel styling."""

    figsize_w: float = 55.0
    figsize_h: float = 55.0
    dpi: int = 300


@dataclass(frozen=True)
class DendrogramVizConfig:
    """Dendrogram panel styling."""

    figsize_w: float = 30.0
    figsize_h: float = 15.0
    dpi: int = 300


@dataclass(frozen=True)
class DecisionPlotVizConfig:
    """Decision plot export settings."""

    dpi: int = 120


@dataclass(frozen=True)
class ForcePlotVizConfig:
    """Force plot export settings."""

    dpi: int = 120
    rounded_figsize_w: float = 20.0
    rounded_figsize_h: float = 3.0


@dataclass(frozen=True)
class TreeVizConfig:
    """Decision tree visualization settings."""

    figsize_w: float = 15.0
    figsize_h: float = 15.0
    dpi: int = 150


@dataclass(frozen=True)
class ConfusionMatrixVizConfig:
    """Confusion matrix visualization settings."""

    figsize_w: float = 10.0
    figsize_h: float = 8.0
    dpi: int = 200


@dataclass(frozen=True)
class RocCurveVizConfig:
    """ROC curve visualization settings."""

    figsize_w: float = 10.0
    figsize_h: float = 8.0
    dpi: int = 200


@dataclass(frozen=True)
class VisualizationConfig:
    """Visualization config container."""

    correlation: CorrelationVizConfig = field(default_factory=CorrelationVizConfig)
    dendrogram: DendrogramVizConfig = field(default_factory=DendrogramVizConfig)
    decision_plot: DecisionPlotVizConfig = field(default_factory=DecisionPlotVizConfig)
    force_plot: ForcePlotVizConfig = field(default_factory=ForcePlotVizConfig)
    tree: TreeVizConfig = field(default_factory=TreeVizConfig)
    confusion_matrix: ConfusionMatrixVizConfig = field(default_factory=ConfusionMatrixVizConfig)
    roc_curve: RocCurveVizConfig = field(default_factory=RocCurveVizConfig)


@dataclass(frozen=True)
class OutputConfig:
    """Output paths for metrics and figures."""

    root_dir: str = "results/reproducible_run"
    metrics_dir: str = "metrics"
    figures_dir: str = "figures"
    logs_dir: str = "logs"

    def root_path(self) -> Path:
        """Return root path object."""
        return Path(self.root_dir)


@dataclass(frozen=True)
class ReproducibilityConfig:
    """Global random seed controls."""

    global_seed: int = 42


@dataclass(frozen=True)
class RunConfig:
    """Root configuration object for reproducible runs."""

    data: DataConfig
    cv: CVConfig = field(default_factory=CVConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    shap: ShapConfig = field(default_factory=ShapConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
