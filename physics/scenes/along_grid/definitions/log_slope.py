"""Log slope: a power law's power, read as the least-squares slope of ln(value) against ln(across).

Contract: `log_slope(across, value)` takes two equal-length sequences of floats and
returns the slope as a float: for value = c * across ** p it is p. NaN with fewer than
2 points or when any number is 0 or below (a power cannot be read there; nothing is
dropped to get one). Raises ValueError for unequal lengths.
"""
from __future__ import annotations

import numpy as np


def log_slope(across, value) -> float:
    """Slope of ln(value) against ln(across); NaN unless there are 2 or more points, all above 0."""
    across, value = np.asarray(across, dtype=np.float64), np.asarray(value, dtype=np.float64)
    if across.shape != value.shape:
        raise ValueError(f"need equal lengths, got {across.shape} and {value.shape}")
    if across.size < 2 or np.any(across <= 0.0) or np.any(value <= 0.0):
        return float("nan")
    return float(np.polyfit(np.log(across), np.log(value), 1)[0])
