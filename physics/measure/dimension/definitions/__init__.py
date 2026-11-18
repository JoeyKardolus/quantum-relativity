"""Dimension, definitions: what things are.

  adjacency.py    adjacency: touching pairs as one symmetric 0/1 sparse matrix
  grid_graph.py   grid_graph: a plain grid of known dimension, the reference
  range_rule.py   range_rule: the shell radii and walk times a graph's size allows
  pass_mark.py    WITHIN, passes: how close to a known dimension a measure must come
"""
from physics.measure.dimension.definitions.adjacency import adjacency
from physics.measure.dimension.definitions.grid_graph import grid_graph
from physics.measure.dimension.definitions.pass_mark import WITHIN, passes
from physics.measure.dimension.definitions.range_rule import range_rule

__all__ = ["adjacency", "grid_graph", "range_rule", "WITHIN", "passes"]
