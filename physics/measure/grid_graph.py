"""Grid graph: a plain square or cubic grid, the known reference for the dimension measures.

Contract: `grid_graph(side, dimensions)` returns (low, high, node_count) for a grid of
side**dimensions nodes, each joined to the nodes one step away along every axis (no
wrap-around). Its true dimension is `dimensions`, so a dimension measure is trusted on
the foam only after it recovers that number here (bible "How results are trusted").
Raises ValueError for side < 2 or dimensions < 1.
"""
from __future__ import annotations

import numpy as np


def grid_graph(side: int, dimensions: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Every neighbouring pair of a side**dimensions grid, and its node count."""
    if side < 2 or dimensions < 1:
        raise ValueError(f"need side >= 2 and dimensions >= 1, got {side} and {dimensions}")
    ids = np.arange(side ** dimensions, dtype=np.int64).reshape((side,) * dimensions)
    lows, highs = [], []
    for axis in range(dimensions):
        lows.append(np.moveaxis(ids, axis, 0)[:-1].ravel())
        highs.append(np.moveaxis(ids, axis, 0)[1:].ravel())
    return np.concatenate(lows), np.concatenate(highs), int(ids.size)
