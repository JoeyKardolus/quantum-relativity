"""Contact pairs: which cells touch which, and across how many box faces (host, numpy).

Contract: `contact_pairs(owner)` takes the owner of every Planck box (int numpy array
of shape (side, side, side), a cell id or -1 for an empty box) and returns three int64
arrays (low, high, faces), one entry per pair of different cells that share at least
one box face: low < high their ids and faces the number of box faces between them.
Two cells touch only across a face; boxes meeting at an edge or a corner do not touch
(bible: joints meet each other only along thin edges). Empty boxes touch nothing.
"""
from __future__ import annotations

import numpy as np


def contact_pairs(owner: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every touching pair of cells (low id, high id) and its shared face count."""
    owner = np.asarray(owner, dtype=np.int64)
    lows, highs = [], []
    for axis in range(3):
        here = np.moveaxis(owner, axis, 0)[:-1].ravel()
        there = np.moveaxis(owner, axis, 0)[1:].ravel()
        touching = (here >= 0) & (there >= 0) & (here != there)
        lows.append(np.minimum(here[touching], there[touching]))
        highs.append(np.maximum(here[touching], there[touching]))
    low = np.concatenate(lows)
    high = np.concatenate(highs)
    if low.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    span = int(owner.max()) + 1
    keys, faces = np.unique(low * span + high, return_counts=True)
    return keys // span, keys % span, faces.astype(np.int64)
