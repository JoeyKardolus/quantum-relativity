"""Deceleration: the astronomer's q, whether a growing size slows (q > 0) or speeds up (q < 0).

Contract: `deceleration(times, sizes)` takes two equal-length sequences of floats (times
rising) and returns q = -(d2a/dt2) a / (da/dt)**2 at the middle of the time span, from a
least-squares quadratic a = c0 + c1 t + c2 t**2 through the samples, a float. q < 0: the
growth speeds up (accelerating expansion); q > 0: it slows; for a = t**p, q = (1 - p) / p
(t**2: -0.5; t**0.5: +1; a straight line: 0). NaN with fewer than 3 samples or da/dt = 0
there. Raises ValueError for unequal lengths or times not rising.
"""
from __future__ import annotations

import numpy as np


def deceleration(times, sizes) -> float:
    """q = -a'' a / a'**2 at mid-span from a quadratic fit of size against time."""
    times = np.asarray(times, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    if times.shape != sizes.shape:
        raise ValueError(f"need equal lengths, got {times.shape} and {sizes.shape}")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must rise")
    if times.size < 3:
        return float("nan")
    middle = (times[0] + times[-1]) / 2.0
    curve, slope, level = np.polyfit(times - middle, sizes, 2)
    if slope == 0.0:
        return float("nan")
    return float(-2.0 * curve * level / slope ** 2)
