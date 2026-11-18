"""Measure dimension: both dimensions of one graph, with the ranges set by the graph's own size.

Contract: `measure_dimension(matrix, sources=8)` takes a symmetric adjacency matrix
(definitions/adjacency.py) and does, in order: picks the `sources` most central nodes
(central_nodes.py); reads the farthest node the first of them can reach, in steps; takes
the radii and times of definitions/range_rule.py; runs ball_dimension.py and
spectral_dimension.py from those sources. Returns a dict of plain numbers and lists:
  ball_dimension, spectral_dimension   the two measured dimensions
  farthest             steps from the first source to the farthest node it reaches
  radii, shells        the shell radii and the mean shell size at each
  times, return_probability   the walk times and the mean chance of being home at each
  nodes                nodes with at least one neighbour
No verdict. Run it on the reference grid of the same size and shape too
(definitions/grid_graph.py) and report both: a small graph pulls both numbers off the
true dimension. Raises what the three measures raise.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from physics.measure.dimension.definitions.range_rule import range_rule
from physics.measure.dimension.logic.ball_dimension import ball_dimension
from physics.measure.dimension.logic.central_nodes import central_nodes
from physics.measure.dimension.logic.spectral_dimension import spectral_dimension


def measure_dimension(matrix: sparse.csr_matrix, sources: int = 8) -> dict:
    """Ball and spectral dimension from the most central nodes, by the range rule."""
    picked = central_nodes(matrix, sources)
    steps = csgraph.shortest_path(matrix, unweighted=True, indices=picked[0])
    farthest = int(steps[np.isfinite(steps)].max())
    radii, times = range_rule(farthest)
    ball = ball_dimension(matrix, picked, radii)
    spectral = spectral_dimension(matrix, picked, times)
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    return {"ball_dimension": ball["dimension"], "spectral_dimension": spectral["dimension"],
            "farthest": farthest, "radii": ball["radii"], "shells": ball["shells"], "times": spectral["times"],
            "return_probability": spectral["return_probability"], "nodes": int(np.count_nonzero(degree > 0))}
