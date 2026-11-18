"""Reference grid: the plain wrapping grid a universe's volume graph is measured beside.

Contract: `reference_grid(volumes_a_side)` returns the symmetric adjacency matrix
(physics.measure adjacency) of a cubic grid of volumes_a_side ** 3 nodes, closed round
on itself as the universe is, every node touching its 26 neighbours: the graph of the
engine's start, where every energy volume has a joint to each of the 26 round it
(axiom 3). Its true dimension is KNOWN_DIMENSION, so what a dimension measure returns
on it is what "3" reads as at this size and shape. Raises what grid_graph raises.
"""
from __future__ import annotations

from scipy import sparse

from physics.measure import adjacency, grid_graph
from physics.scenes.dimension.definitions.known import KNOWN_DIMENSION


def reference_grid(volumes_a_side: int) -> sparse.csr_matrix:
    """The wrapping 26-neighbour cubic grid of that many nodes a side, as an adjacency matrix."""
    return adjacency(*grid_graph(int(volumes_a_side), KNOWN_DIMENSION, wrapped=True, diagonals=True))
