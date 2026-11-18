"""Orbit kick: the sideways kick that lets a lump go round at its distance.

Contract: two formulas, plain floats.
  fall_rate(pull_step, step_time)
      how fast the pull speeds a lump up, in boxes a step squared: `2 step_time ** 2 *
      abs(pull_step)`, `pull_step` being the pull's rise from one box to the next. It is
      the engine's own fluid law: a phase step `k` moves fluid `2 step_time sin k`
      boxes a step (pair_rotation.py) and the pull turns the phase by `pull * step_time`
      a step (pull_fluid.py), on a clock at full rate.
  orbit_kick(pull_step, distance, step_time)
      the phase step, radians a box, of the speed `sqrt(fall_rate * distance)` a circle
      of radius `distance` boxes asks for: `asin(speed / (SWIRL_SPEED * 2 step_time))`,
      at most pi / 2. `SWIRL_SPEED` (cos 30 degrees) is what a standing swirl's fluid
      keeps of a kick's speed: its own waves run at +-30 degrees a box.
"""
from __future__ import annotations

import math

SWIRL_SPEED = math.cos(math.pi / 6.0)


def fall_rate(pull_step: float, step_time: float) -> float:
    """Boxes a step squared the pull adds to a lump's speed."""
    return 2.0 * step_time ** 2 * abs(pull_step)


def orbit_kick(pull_step: float, distance: float, step_time: float) -> float:
    """The kick, radians a box, of a circle of `distance` boxes in a pull rising `pull_step` a box."""
    speed = math.sqrt(fall_rate(pull_step, step_time) * distance)
    return math.asin(min(speed / (SWIRL_SPEED * 2.0 * step_time), 1.0))
