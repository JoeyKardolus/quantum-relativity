"""Ball dimension: the Hausdorff dimension of a graph from how fast balls of steps grow.

Contract: `ball_dimension(matrix, sources, radii)` takes a symmetric adjacency matrix
(definitions/adjacency.py), source node ids and whole-step radii (at least two, all >= 1). From
each source it counts the nodes exactly r steps away (the shell), averages the shells
over the sources, and fits ln(shell) against ln(r): in d dimensions a ball grows as
r**d, so its shell as r**(d - 1), and the dimension is 1 + that slope. Returns a dict:
  dimension   1 + the least-squares slope over the radii
  radii       the radii (ints); shells  the mean shell size at each
Choose radii well inside the graph: a ball that reaches the edge stops growing.
Raises ValueError for fewer than two radii, a radius < 1, no source, or an empty shell.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph


def ball_dimension(matrix: sparse.csr_matrix, sources, radii) -> dict:
    """1 + the slope of ln(shell size) against ln(radius), shells averaged over the sources."""
    sources = np.atleast_1d(np.asarray(sources, dtype=np.int64))
    radii = np.asarray(radii, dtype=np.int64)
    if sources.size == 0 or radii.size < 2 or radii.min() < 1:
        raise ValueError(f"need sources and at least two radii >= 1, got {sources.size} sources, radii {radii}")
    steps = csgraph.shortest_path(matrix, unweighted=True, indices=sources)
    shells = np.array([np.mean(np.sum(steps == radius, axis=1)) for radius in radii])
    if np.any(shells == 0):
        raise ValueError(f"an empty shell at radii {radii[shells == 0].tolist()}: the graph ends there")
    slope, _ = np.polyfit(np.log(radii), np.log(shells), 1)
    return {"dimension": float(1.0 + slope), "radii": radii.tolist(), "shells": shells.tolist()}
