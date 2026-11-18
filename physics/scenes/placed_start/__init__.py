"""Placed start: an even grid in which chosen energy volumes hold more energy.

  definitions/  place_cell (a lattice place's cell id), even_fill (whole quanta spread
                evenly over a block)
  logic/        packed_start (the universe)
"""
from physics.scenes.placed_start.definitions.even_fill import even_fill
from physics.scenes.placed_start.definitions.place_cell import place_cell
from physics.scenes.placed_start.logic.packed_start import packed_start

__all__ = ["even_fill", "packed_start", "place_cell"]
