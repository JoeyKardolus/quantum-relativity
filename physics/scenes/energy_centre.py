"""Energy centre: the energy-weighted centre of the boxes a mask picks.

Contract: `energy_centre(energy, mask=None)` takes a float array of energy per box,
shape (side, side, side), and an optional boolean mask of the same shape (None: every
box). Returns the energy-weighted mean box index (x, y, z) as a float64 array of 3, in
box units with box i's centre at i. Returns NaN in all three when the picked boxes
hold no energy.
"""
from __future__ import annotations

import numpy as np


def energy_centre(energy: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Energy-weighted mean box index of the masked boxes; NaN when they hold nothing."""
    weight = energy if mask is None else np.where(mask, energy, 0.0)
    total = float(weight.sum())
    if total <= 0.0:
        return np.full(3, np.nan)
    index = np.arange(energy.shape[0], dtype=np.float64)
    return np.array([float((weight.sum(axis=(1, 2)) * index).sum()),
                     float((weight.sum(axis=(0, 2)) * index).sum()),
                     float((weight.sum(axis=(0, 1)) * index).sum())]) / total
