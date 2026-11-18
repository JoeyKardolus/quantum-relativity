"""Spin scene: does a cell stay the one whole swirl it started as while every rule runs?

  definitions/  spin_cells (which cells are watched), turn_record (what a cell's
                readings say), the pass mark
  logic/        spin (the scene)
"""
from physics.scenes.spin.definitions.pass_mark import NOT_ASKED, passes
from physics.scenes.spin.definitions.spin_cells import spin_cells
from physics.scenes.spin.definitions.turn_record import turn_record
from physics.scenes.spin.logic.spin import spin

__all__ = ["NOT_ASKED", "passes", "spin", "spin_cells", "turn_record"]
