"""Measures: numbers read from the foam's relations alone, each validated on a known reference first.

Bible `physics/quantum-relativity`, rule 1 and "How results are trusted": distance and
dimension come from which cells touch (neighbour counts, random walks), and a measure
is trusted on the foam only after it gives the right answer on a reference whose true
value is known (plain 2D and 3D grids; a placed whirl of known turns). Host numpy and
scipy only; no engine import, so every measure runs without a GPU.

Module map:
  adjacency.py           adjacency: touching pairs as one symmetric 0/1 sparse matrix
  grid_graph.py          grid_graph: a plain square or cubic grid, the dimension reference
  central_nodes.py       central_nodes: the nodes deepest inside a graph, from far points
  ball_dimension.py      ball_dimension: Hausdorff dimension from shell growth round sources
  spectral_dimension.py  spectral_dimension: dimension from a lazy random walk's return chance
  phase_winding.py       phase_winding: whole turns of a complex field's phase round every unit square
  whirl_line.py          whirl_line: a whirl's centre, direction and length from the pierced squares
  loop_circulation.py    loop_circulation: winding and flow circulation round a square loop
  whirl_strength.py      whirl_strength: angular momentum about a line per unit density
"""
from physics.measure.adjacency import adjacency
from physics.measure.grid_graph import grid_graph
from physics.measure.central_nodes import central_nodes
from physics.measure.ball_dimension import ball_dimension
from physics.measure.spectral_dimension import spectral_dimension
from physics.measure.phase_winding import phase_winding
from physics.measure.whirl_line import whirl_line
from physics.measure.loop_circulation import loop_circulation
from physics.measure.whirl_strength import whirl_strength

__all__ = ["adjacency", "grid_graph", "central_nodes", "ball_dimension", "spectral_dimension", "phase_winding",
           "whirl_line", "loop_circulation", "whirl_strength"]
