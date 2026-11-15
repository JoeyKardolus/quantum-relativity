"""Central nodes: the nodes deepest inside a graph, found from the graph alone (no coordinates).

Contract: `central_nodes(matrix, count, peripherals=16)` takes a symmetric adjacency
matrix (adjacency.py) and returns `count` node ids (int64) of its largest connected
piece, deepest first. Depth comes from far points: start at the piece's first node,
then add `peripherals` times the node farthest (in steps) from every far point chosen so
far; a node's score is its largest distance to a far point (ties: the smaller sum), the
smallest score first. On a grid the far points are its corners and the winners its
middle. Raises ValueError for count < 1, peripherals < 2, or a piece smaller than count.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph


def central_nodes(matrix: sparse.csr_matrix, count: int, peripherals: int = 16) -> np.ndarray:
    """`count` nodes of the largest connected piece, most central first."""
    if count < 1 or peripherals < 2:
        raise ValueError(f"need count >= 1 and peripherals >= 2, got {count} and {peripherals}")
    _, labels = csgraph.connected_components(matrix, directed=False)
    largest = np.bincount(labels).argmax()
    piece = np.flatnonzero(labels == largest)
    if piece.size < count:
        raise ValueError(f"the largest connected piece has {piece.size} nodes, fewer than {count}")
    far_points = [int(piece[0])]
    distances = []
    for _ in range(peripherals):
        distances.append(csgraph.shortest_path(matrix, unweighted=True, indices=far_points[-1])[piece])
        nearest = np.min(distances, axis=0)
        far_points.append(int(piece[np.argmax(nearest)]))
    rows = np.array(distances[1:])
    order = np.lexsort((rows.sum(axis=0), rows.max(axis=0)))
    return piece[order[:count]].astype(np.int64)
