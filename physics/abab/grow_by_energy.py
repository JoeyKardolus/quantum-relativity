"""Grow by energy: an energy volume's size follows its energy (bible foundation 3: more energy, smaller).

Contract: `grow_by_energy(size, before, after)` returns size * (before / after) ** 3
elementwise: the energy-volume law size ~ energy ** -3 at fixed content (the cells
engine's rest_volume q (q / E) ** 3 with its quanta q held), applied to a change of
energy, so an energy volume that gains gets smaller and more packed and one that
loses grows. The content is not stored: it is what the size and energy already say.
A volume whose energy did not change keeps its size exactly.
"""
from __future__ import annotations

import numpy as np


def grow_by_energy(size: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Size times (energy before / energy after) cubed."""
    return size * (before / after) ** 3
