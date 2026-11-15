"""Start reference: the start's mean density of each cell kind, and how many boxes deep it is.

Contract: `start_reference(density, kind)` takes the start's per-box energy density
of the occupied boxes and the kind of each box's cell (the view's `density` and
`kind`, ENERGY_VOLUME 0 or JOINT 1: numpy, or anything with `.numpy()` such as a
Warp array) and returns a float64 array of three values: index 0 the mean density
of the energy-volume boxes, index 1 the mean density of the joint boxes (each 1.0
when that kind has no box or no finite positive mean), index 2 the boxes along one
line of sight, the cube root of the occupied box count of both kinds (at least 1).
Each kind is lit against its own mean, so joints, which hold less energy per box
than energy volumes, are as visible as they are. Taken once at the start (and again
on reset) and then held fixed, so contrast reads a box against where the energy of
its kind began, never against the current spread, and a growing universe is not
dimmed by its own growth. Copies the arrays to the host once; meant for resets
only. Raises ValueError when the two arrays differ in length.
"""
from __future__ import annotations

import numpy as np

_KINDS = (0, 1)


def start_reference(density, kind) -> np.ndarray:
    """The mean density of each kind's boxes, and the occupied count's cube root."""
    energies = _host(density).astype(np.float64).ravel()
    kinds = _host(kind).astype(np.int64).ravel()
    if energies.size != kinds.size:
        raise ValueError(f"{energies.size} densities but {kinds.size} kinds")
    reference = np.ones(3)
    for slot, wanted in enumerate(_KINDS):
        chosen = energies[kinds == wanted]
        if chosen.size:
            mean = float(chosen.mean())
            reference[slot] = mean if np.isfinite(mean) and mean > 0 else 1.0
    reference[2] = max(1.0, energies.size ** (1.0 / 3.0))
    return reference


def _host(array) -> np.ndarray:
    return array.numpy() if hasattr(array, "numpy") else np.asarray(array)
