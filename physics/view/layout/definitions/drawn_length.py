"""Drawn length: the length a cell is drawn with.

Contract: `drawn_length(length, alive)` takes every cell's length by the engine's size
law and whether it lives, and returns float64 lengths to draw: a living cell's own
length; 0 for a cell that is gone. A living cell holding no energy is endlessly long by
the size law (`inf`) and cannot be drawn so: it is drawn as long as the longest living
cell that has a length (1 when none has). That one case is the only length here that
is not the engine's.
"""
from __future__ import annotations

import numpy as np


def drawn_length(length: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Every cell's length as drawn: its own, 0 when gone, the longest finite one when endless."""
    length = np.asarray(length, dtype=np.float64)
    alive = np.asarray(alive, dtype=bool)
    finite = alive & np.isfinite(length)
    longest = float(length[finite].max()) if finite.any() else 1.0
    return np.where(alive, np.where(np.isfinite(length), length, longest), 0.0)
