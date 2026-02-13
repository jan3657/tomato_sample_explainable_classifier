"""Decision-tree rendering helper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn import tree


def plot_last_fold_tree(
    dt_model: object,
    feature_names: list[str],
    class_names: list[str],
    out_path: str,
    figsize_w: float,
    figsize_h: float,
    dpi: int,
) -> str:
    """Plot the decision tree fitted on the final cross-validation fold."""
    plt.figure(figsize=(figsize_w, figsize_h))
    tree.plot_tree(dt_model, feature_names=feature_names, class_names=class_names, filled=True)
    plt.title("Decision Tree (Final Fold)")
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close()
    return str(output)
