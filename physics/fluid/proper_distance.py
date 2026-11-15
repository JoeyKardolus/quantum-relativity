"""Proper distance: how far every occupied box is from one box, measured along the structure.

Contract: `proper_distance(length, occupied, origin)` takes numpy arrays per box of
local length (local_length.py) and a bool mask of the boxes that are space (they
hold energy), and an origin index (x, y, z) that must be occupied; it returns an
array of the shortest path length from the origin to each box, stepping only between
occupied face neighbours, each step costing the mean of the two boxes' lengths (inf
for a box not reached). Bible rule 1: distance comes from the grid's relations and
the sizes energy gives, never from coordinates. Host-side (scipy Dijkstra): meant
for measurements, seconds at side 64. Raises ValueError when the origin is not
occupied.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def proper_distance(length: np.ndarray, occupied: np.ndarray, origin: tuple[int, int, int]) -> np.ndarray:
    """Shortest summed half-lengths path from the origin to every occupied box."""
    if not occupied[origin]:
        raise ValueError(f"origin {origin} holds no energy: it is not part of space")
    index = np.arange(occupied.size).reshape(occupied.shape)
    starts, ends, costs = [], [], []
    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis], high[axis] = slice(0, -1), slice(1, None)
        both = occupied[tuple(low)] & occupied[tuple(high)]
        starts.append(index[tuple(low)][both])
        ends.append(index[tuple(high)][both])
        costs.append(0.5 * (length[tuple(low)][both] + length[tuple(high)][both]))
    rows, cols, weights = (np.concatenate(parts) for parts in (starts, ends, costs))
    graph = coo_matrix((weights, (rows, cols)), shape=(occupied.size, occupied.size)).tocsr()
    distance = dijkstra(graph, directed=False, indices=int(index[origin]))
    return distance.reshape(occupied.shape)
