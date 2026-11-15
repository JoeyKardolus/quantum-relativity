"""Joint balance: the thickness where a joint's push cancels the squeeze on it.

Contract: `joint_balance(grid, settings)` returns one thickness per joint (a, b): the
point where swing_joints()'s force is zero with the grid held as it is,
  joint_push * (R - t) = s * t   ->   t = k R / (k + s),
k = joint_push, s = attraction * (p_a + p_b) / 2 (p = pull_potential(grid)),
R = joint_rest(volume, joints, joint_rest). R when k + s is 0. Reads only the Grid.
new_engine() starts every joint here at rest rate 0; it swings from there as the
energy around it moves.
"""
from __future__ import annotations

import numpy as np

from physics.cells.engine_settings import EngineSettings
from physics.cells.grid import Grid
from physics.cells.joint_rest import joint_rest
from physics.cells.pull_potential import pull_potential


def joint_balance(grid: Grid, settings: EngineSettings) -> np.ndarray:
    """Per joint, the thickness where push and squeeze cancel: k R / (k + s)."""
    exerted = settings.attraction * pull_potential(grid) / 2
    squeeze = exerted[grid.joints[:, 0]] + exerted[grid.joints[:, 1]]
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    held = settings.joint_push + squeeze
    return np.divide(settings.joint_push * rest, held, out=rest.copy(), where=held > 0)
