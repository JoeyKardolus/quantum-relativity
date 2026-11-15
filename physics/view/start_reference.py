"""Start reference: the start's mean energy per occupied box, per kind, the fixed unit of contrast.

Contract: `start_reference(a_energy, b_energy)` takes the two per-box energy arrays
(numpy, or anything with `.numpy()` such as a Warp array; any shape, same size)
and returns a float64 array of two values: index 0 the mean pulling (A) energy and
index 1 the mean pushing (B) energy, each over the boxes holding any energy at all
(A + B > 0). Taken once at the start (and again on reset) and then held fixed, so
contrast reads a box against where the energy began, never against the current
spread. A kind with no occupied box, or no finite positive mean, gets 1.0. Each
kind has its own unit so a kind that starts with little energy is not invisible
from the first step. Copies both arrays to the host once; meant for resets only.
"""
from __future__ import annotations

import numpy as np


def start_reference(a_energy, b_energy) -> np.ndarray:
    """The mean A and B energy over the occupied boxes."""
    pulling = _host(a_energy).astype(np.float64).ravel()
    pushing = _host(b_energy).astype(np.float64).ravel()
    occupied = (pulling + pushing) > 0
    reference = np.ones(2)
    if occupied.any():
        for which, energies in enumerate((pulling, pushing)):
            mean = float(energies[occupied].mean())
            reference[which] = mean if np.isfinite(mean) and mean > 0 else 1.0
    return reference


def _host(array) -> np.ndarray:
    return array.numpy() if hasattr(array, "numpy") else np.asarray(array)
