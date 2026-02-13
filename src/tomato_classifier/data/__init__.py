"""Data loading and fold split helpers."""

from .io import build_xy, load_dataset
from .splits import FoldIndices, generate_stratified_folds

__all__ = ["load_dataset", "build_xy", "FoldIndices", "generate_stratified_folds"]
