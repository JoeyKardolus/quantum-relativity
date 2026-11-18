"""Timescape scene: does growth look different timed by clocks in the most packed volumes?

  definitions/  deceleration (the astronomer's q), KNOWN_Q, the pass mark
  logic/        timescape (the scene)
"""
from physics.scenes.timescape.definitions.deceleration import deceleration
from physics.scenes.timescape.definitions.known import KNOWN_Q
from physics.scenes.timescape.definitions.pass_mark import GREW, LOWER_BY, passes
from physics.scenes.timescape.logic.timescape import timescape

__all__ = ["GREW", "KNOWN_Q", "LOWER_BY", "deceleration", "passes", "timescape"]
