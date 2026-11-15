"""Growth rates: how fast a size grows, d ln(size) / dt, between consecutive samples.

Contract: `growth_rates(times, sizes)` takes two equal-length sequences of floats (times
rising, sizes > 0) and returns (midpoints, rates), two float64 arrays one shorter: rate
i is (ln size[i+1] - ln size[i]) / (time[i+1] - time[i]) at the midpoint of the two
times, the Hubble-like rate (per time unit; positive = growing). Raises ValueError for
fewer than 2 samples, unequal lengths, times not rising or a size <= 0.
"""
from __future__ import annotations

import numpy as np


def growth_rates(times, sizes) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint times and d ln(size)/dt between consecutive samples."""
    times = np.asarray(times, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)
    if times.size < 2 or times.shape != sizes.shape:
        raise ValueError(f"need at least 2 samples of equal length, got {times.shape} and {sizes.shape}")
    if np.any(np.diff(times) <= 0) or np.any(sizes <= 0):
        raise ValueError("times must rise and sizes must be > 0")
    return (times[1:] + times[:-1]) / 2.0, np.diff(np.log(sizes)) / np.diff(times)
