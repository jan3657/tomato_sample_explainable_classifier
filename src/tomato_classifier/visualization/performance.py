"""Performance-plot helpers for classification outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    out_path: str,
    figsize_w: float,
    figsize_h: float,
    dpi: int,
) -> str:
    """Plot and save confusion matrix for class predictions."""
    labels = [str(label) for label in class_labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h), dpi=dpi)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (Best Model)")
    plt.tight_layout()

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def plot_multiclass_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_labels: list[str],
    out_path: str,
    figsize_w: float,
    figsize_h: float,
    dpi: int,
) -> str:
    """Plot one-vs-rest ROC curves for each class plus micro-average."""
    labels = [str(label) for label in class_labels]
    y_true_bin = label_binarize(y_true, classes=labels)
    if y_true_bin.ndim == 1:
        y_true_bin = y_true_bin.reshape(-1, 1)

    if len(labels) == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.column_stack((1 - y_true_bin[:, 0], y_true_bin[:, 0]))

    scores = np.asarray(y_score, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    if len(labels) == 2 and scores.shape[1] == 1:
        scores = np.column_stack((1 - scores[:, 0], scores[:, 0]))

    if scores.shape[1] != len(labels):
        raise ValueError(
            "ROC scores must have one column per class label. "
            f"Expected {len(labels)}, got {scores.shape[1]}."
        )

    fig, ax = plt.subplots(figsize=(figsize_w, figsize_h), dpi=dpi)
    plotted_any_curve = False
    for class_index, class_label in enumerate(labels):
        y_class = y_true_bin[:, class_index]
        if np.unique(y_class).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_class, scores[:, class_index])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{class_label} (AUC = {roc_auc:.3f})")
        plotted_any_curve = True

    if y_true_bin.size > 0 and np.unique(y_true_bin.ravel()).size > 1:
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), scores.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(
            fpr_micro,
            tpr_micro,
            color="black",
            linestyle="--",
            lw=2,
            label=f"micro-average (AUC = {auc_micro:.3f})",
        )
        plotted_any_curve = True

    if not plotted_any_curve:
        plt.close(fig)
        raise ValueError("ROC curve could not be plotted: labels are degenerate.")

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (Best Model)")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(output)
