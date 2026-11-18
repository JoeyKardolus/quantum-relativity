"""Grid graph: a plain square or cubic grid, the known reference for the dimension measures.

Contract: `grid_graph(side, dimensions, wrapped=False, diagonals=False)` returns
(low, high, node_count) for a grid of side**dimensions nodes, node (i, j, ...) numbered
in raster order (the last coordinate runs fastest).
  diagonals=False  each node touches the nodes one step away along one axis
                   (4 neighbours in 2D, 6 in 3D)
  diagonals=True   it also touches the nodes one step away along several axes at once
                   (8 in 2D, 26 in 3D): the engine's start, where every volume has a
                   joint to each of its 26 neighbours (axiom 3)
  wrapped=False    the grid ends at its faces
  wrapped=True     it closes round on itself, as the universe does
Its true dimension is `dimensions`, so a dimension measure is trusted on the engine's
grid only after it recovers that number here. On a wrapped grid of side 2 a pair is
listed more than once (adjacency.py counts it once). Raises ValueError for side < 2 or
dimensions < 1.
"""
from __future__ import annotations

from itertools import product

import numpy as np


def _half_offsets(dimensions: int, diagonals: bool) -> list:
    """One of every two opposite one-step offsets: those whose first nonzero part is +1."""
    offsets = []
    for offset in product((-1, 0, 1), repeat=dimensions):
        moved = [part for part in offset if part != 0]
        if moved and moved[0] == 1 and (diagonals or len(moved) == 1):
            offsets.append(offset)
    return offsets


def _open_pairs(ids: np.ndarray, offset: tuple) -> tuple[np.ndarray, np.ndarray]:
    """The pairs one offset apart in a grid that ends at its faces."""
    here = tuple({1: slice(None, -1), -1: slice(1, None), 0: slice(None)}[part] for part in offset)
    there = tuple({1: slice(1, None), -1: slice(None, -1), 0: slice(None)}[part] for part in offset)
    return ids[here].ravel(), ids[there].ravel()


def _wrapped_pairs(ids: np.ndarray, offset: tuple) -> tuple[np.ndarray, np.ndarray]:
    """The pairs one offset apart in a grid closed round on itself."""
    there = np.roll(ids, shift=[-part for part in offset], axis=tuple(range(ids.ndim)))
    return ids.ravel(), there.ravel()


def grid_graph(side: int, dimensions: int, wrapped: bool = False,
               diagonals: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
    """Every neighbouring pair of a side**dimensions grid, and its node count."""
    if side < 2 or dimensions < 1:
        raise ValueError(f"need side >= 2 and dimensions >= 1, got {side} and {dimensions}")
    ids = np.arange(side ** dimensions, dtype=np.int64).reshape((side,) * dimensions)
    pairs_of = _wrapped_pairs if wrapped else _open_pairs
    pairs = [pairs_of(ids, offset) for offset in _half_offsets(dimensions, diagonals)]
    return np.concatenate([low for low, _ in pairs]), np.concatenate([high for _, high in pairs]), int(ids.size)
