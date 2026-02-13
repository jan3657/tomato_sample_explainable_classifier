"""Feature clustering and selection routines."""

from .artifacts import FoldFeatureSelectionResult
from .clustering import (
    compute_abs_correlation,
    compute_linkage_from_correlation,
    select_representatives,
    select_features_for_fold,
)

__all__ = [
    "FoldFeatureSelectionResult",
    "compute_abs_correlation",
    "compute_linkage_from_correlation",
    "select_representatives",
    "select_features_for_fold",
]
