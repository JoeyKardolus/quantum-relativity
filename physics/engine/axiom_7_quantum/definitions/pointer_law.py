"""Pointer law: where a split's pointer jumps, with each cell's share of the energy as its chance.

Contract: `pointer_law(part_source, part_cell, part_quanta, sources, uniform)` (host
numpy) takes the halves' table (in rising source, then cell order), the splits to draw
for (`sources`, each with at least one half that holds quanta) and one draw in [0, 1)
per split. It returns one cell per split: a half's cell, its chance its quanta over the
split's quanta (the draw times the split's quanta, floored, is walked through the halves
in cell order). A half of 0 is never chosen. Raises ValueError for a split with no quanta.
"""
from __future__ import annotations

import numpy as np


def pointer_law(part_source, part_cell, part_quanta, sources, uniform) -> np.ndarray:
    """The cell each split's pointer jumps to, drawn by share."""
    sources = np.asarray(sources, dtype=np.int64)
    if len(sources) == 0:
        return np.zeros(0, dtype=np.int64)
    running = np.cumsum(part_quanta, dtype=np.int64)
    first = np.searchsorted(part_source, sources, side="left")
    last = np.searchsorted(part_source, sources, side="right")
    before = np.where(first > 0, running[np.maximum(first, 1) - 1], 0)
    amount = np.where(last > 0, running[np.maximum(last, 1) - 1], 0) - before
    if np.any(amount <= 0):
        raise ValueError("a pointer needs a split that holds quanta")
    drawn = np.minimum((np.asarray(uniform, dtype=np.float64) * amount).astype(np.int64), amount - 1)
    return np.asarray(part_cell)[np.searchsorted(running, before + drawn, side="right")]
