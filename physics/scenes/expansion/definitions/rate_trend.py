"""Rate trend: whether a rate rises or falls over a stretch of time.

Contract: `rate_trend(times, rates)` takes two equal-length sequences of floats and
returns the least-squares slope of rate against time as a float: below 0 the rate
falls (growth slows), above 0 it rises (growth speeds up). NaN with fewer than 2
points. Raises ValueError for unequal lengths.
"""
from __future__ import annotations

import numpy as np


def rate_trend(times, rates) -> float:
    """Slope of a straight line through rate against time; NaN with fewer than 2 points."""
    times, rates = np.asarray(times, dtype=np.float64), np.asarray(rates, dtype=np.float64)
    if times.shape != rates.shape:
        raise ValueError(f"need equal lengths, got {times.shape} and {rates.shape}")
    if times.size < 2:
        return float("nan")
    return float(np.polyfit(times, rates, 1)[0])
