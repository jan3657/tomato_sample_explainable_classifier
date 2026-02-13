"""Random-seed setup utilities."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set deterministic seed for Python and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
