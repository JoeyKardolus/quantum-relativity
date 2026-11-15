"""RMS radius: how far, on average, the picked energy sits from its own centre.

Contract: `rms_radius(energy, mask=None)` returns the square root of the energy-weighted
mean squared distance (in boxes) of the masked boxes from their energy centre
(energy_centre.py), a float. 0 for one box; NaN when the picked boxes hold no energy.
For an even ball of radius R it is R * sqrt(3/5).
"""
from __future__ import annotations

import numpy as np

from physics.scenes.energy_centre import energy_centre


def rms_radius(energy: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Energy-weighted RMS distance of the masked boxes from their energy centre."""
    weight = energy if mask is None else np.where(mask, energy, 0.0)
    centre = energy_centre(weight)
    if np.isnan(centre).any():
        return float("nan")
    axis = np.arange(energy.shape[0], dtype=np.float64)
    squared = ((axis[:, None, None] - centre[0]) ** 2 + (axis[None, :, None] - centre[1]) ** 2
               + (axis[None, None, :] - centre[2]) ** 2)
    return float(np.sqrt((weight * squared).sum() / weight.sum()))
