"""Bound lump scene: does the fullest energy volume keep its size while the rest of space changes?

  definitions/  second_half_change, the pass mark
  logic/        bound_lump (the scene)
"""
from physics.scenes.bound_lump.definitions.pass_mark import KEEPS_WITHIN, REST_GROWS, passes
from physics.scenes.bound_lump.definitions.second_half_change import second_half_change
from physics.scenes.bound_lump.logic.bound_lump import bound_lump

__all__ = ["KEEPS_WITHIN", "REST_GROWS", "bound_lump", "passes", "second_half_change"]
