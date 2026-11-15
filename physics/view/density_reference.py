"""Density reference: the start's mean energy density per kind, the fixed unit of opacity.

Contract: `density_reference(kind, density)` returns a float64 array of two values,
index KIND_A (energy volumes) and KIND_B (joints): the mean density of that kind
among the given volumes. Taken once at the start (and again on reset) and then held
fixed, so opacity reads a density against where it began, never against the
current spread. A kind with no volumes, or no finite positive mean, gets 1.0.
Each kind has its own unit because a joint packs its small energy into a thin wall
and would otherwise read as solid from the first step.
"""
from __future__ import annotations

import numpy as np

from physics.abab.state import KIND_A, KIND_B


def density_reference(kind: np.ndarray, density: np.ndarray) -> np.ndarray:
    """The mean density of each kind, indexed by kind."""
    kind = np.asarray(kind)
    density = np.asarray(density, dtype=float)
    reference = np.ones(2)
    for which in (KIND_A, KIND_B):
        chosen = density[kind == which]
        mean = float(chosen.mean()) if chosen.size else 1.0
        reference[which] = mean if np.isfinite(mean) and mean > 0 else 1.0
    return reference
