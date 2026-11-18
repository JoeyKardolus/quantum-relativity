"""Early and late: the mean growth rate over the first and over the last quarter of a run.

Contract: `early_late(rates)` takes a sequence of growth rates in time order and returns
(early, late), two floats: the mean of the first quarter of the samples and of the last
quarter (len // 4 each, at least one). Raises ValueError for no samples.
"""
from __future__ import annotations

import numpy as np


def early_late(rates) -> tuple[float, float]:
    """Mean rate over the first quarter and over the last quarter of the samples."""
    rates = np.asarray(rates, dtype=np.float64)
    if rates.size < 1:
        raise ValueError("need at least one growth rate")
    quarter = max(rates.size // 4, 1)
    return float(rates[:quarter].mean()), float(rates[-quarter:].mean())
