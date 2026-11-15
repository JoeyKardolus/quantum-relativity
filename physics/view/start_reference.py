"""Start reference: the start's mean density per occupied box and how many boxes deep it is.

Contract: `start_reference(density)` takes the start's per-box energy density of the
occupied boxes (the view's `density`: numpy, or anything with `.numpy()` such as a
Warp array) and returns a float64 array of two values: index 0 the mean density
(1.0 when there is no box or no finite positive mean), index 1 the boxes along one
line of sight, the cube root of the occupied box count (at least 1). Taken once at
the start (and again on reset) and then held fixed, so contrast reads a box against
where the energy began, never against the current spread, and a growing universe
is not dimmed by its own growth. Copies the array to the host once; meant for
resets only.
"""
from __future__ import annotations

import numpy as np


def start_reference(density) -> np.ndarray:
    """The mean density over the occupied boxes, and their count's cube root."""
    energies = _host(density).astype(np.float64).ravel()
    reference = np.ones(2)
    if energies.size:
        mean = float(energies.mean())
        reference[0] = mean if np.isfinite(mean) and mean > 0 else 1.0
        reference[1] = max(1.0, energies.size ** (1.0 / 3.0))
    return reference


def _host(array) -> np.ndarray:
    return array.numpy() if hasattr(array, "numpy") else np.asarray(array)
