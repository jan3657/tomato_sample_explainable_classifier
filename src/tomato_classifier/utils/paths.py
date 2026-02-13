"""Output-path preparation helpers."""

from __future__ import annotations

from pathlib import Path

from tomato_classifier.config.schema import OutputConfig


def ensure_output_dirs(output_cfg: OutputConfig) -> dict[str, Path]:
    """Create output root and standard subdirectories."""
    root = Path(output_cfg.root_dir)
    metrics = root / output_cfg.metrics_dir
    figures = root / output_cfg.figures_dir
    logs = root / output_cfg.logs_dir

    for path in (root, metrics, figures, logs):
        path.mkdir(parents=True, exist_ok=True)

    return {"root": root, "metrics": metrics, "figures": figures, "logs": logs}
