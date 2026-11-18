"""Graph dimension: the ball and the spectral dimension of one universe's volume graph.

Contract: `graph_dimension(universe, sources=8)` reads which living energy volumes have
a way to which (physics.engine read_ways and read_cells; physics.measure volume_graph)
and returns physics.measure measure_dimension of that graph: a dict with
ball_dimension, spectral_dimension, farthest, radii, shells, times, return_probability
and nodes. Only relations are read (axiom 3: no coordinates). Writes nothing. Raises
what the measure raises, as when too few volumes are left.
"""
from __future__ import annotations

from physics.engine import Grid, read_cells, read_ways
from physics.measure import measure_dimension, volume_graph


def graph_dimension(universe: Grid, sources: int = 8) -> dict:
    """Both dimensions of the graph of energy volumes joined by ways."""
    matrix, _ = volume_graph(read_ways(universe), read_cells(universe)["kind"])
    return measure_dimension(matrix, sources)
