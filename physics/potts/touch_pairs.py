"""Touch pairs: which cells touch which across a face, an edge or a corner of their boxes (host, numpy).

Contract: `touch_pairs(owner)` takes the owner of every Planck box (int numpy array of
shape (sx, sy, sz), a cell id or -1 for an empty box) and returns two int64 arrays
(low, high), one entry per pair of different owners with a box of one among the 26
neighbours (face, edge, corner) of a box of the other, low < high, sorted, each pair
once. Empty space counts as owner -1, so a cell touching empty space gives a pair
(-1, cell); the boxes beyond the array's faces are not looked at. This is the touch of
the wrapping rule (bible foundation 1: an energy volume touches no other energy volume,
not on a face, an edge or a corner); contact_pairs.py counts the face contacts only.
"""
from __future__ import annotations

import numpy as np

_HALF_OF_26 = [(di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)
               if (di, dj, dk) > (0, 0, 0)]


def touch_pairs(owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every pair of owners (low id, high id) whose boxes meet at a face, an edge or a corner."""
    owner = np.asarray(owner, dtype=np.int64)
    lows, highs = [], []
    for offset in _HALF_OF_26:
        here = owner[tuple(slice(max(0, -step), owner.shape[axis] - max(0, step)) for axis, step in enumerate(offset))]
        there = owner[tuple(slice(max(0, step), owner.shape[axis] - max(0, -step)) for axis, step in enumerate(offset))]
        differ = here != there
        lows.append(np.minimum(here[differ], there[differ]))
        highs.append(np.maximum(here[differ], there[differ]))
    low = np.concatenate(lows)
    high = np.concatenate(highs)
    if low.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()
    span = int(owner.max()) + 2
    keys = np.unique((low + 1) * span + (high + 1))
    return keys // span - 1, keys % span - 1
