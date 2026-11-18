"""Dimension scene: how many dimensions the grid of energy volumes has, beside a plain grid's.

  definitions/  KNOWN_DIMENSION, reference_grid (the plain wrapping grid), the pass mark
  logic/        graph_dimension (one universe measured), space_dimension (the scene)
"""
from physics.scenes.dimension.definitions.known import KNOWN_DIMENSION
from physics.scenes.dimension.definitions.pass_mark import passes
from physics.scenes.dimension.definitions.reference_grid import reference_grid
from physics.scenes.dimension.logic.graph_dimension import graph_dimension
from physics.scenes.dimension.logic.space_dimension import space_dimension

__all__ = ["KNOWN_DIMENSION", "graph_dimension", "passes", "reference_grid", "space_dimension"]
