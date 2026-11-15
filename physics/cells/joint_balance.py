"""Joint balance: the thickness where a joint's push cancels the squeeze on it.

Contract: `joint_balance(grid, settings)` returns one thickness per joint (a, b): the
point where swing_joints()'s force is zero with the grid held as it is,
  joint_push(t, R, k, power) = s * t,   i.e.   k ((R/t)^power - 1) / power = s t / R
k = joint_push, power = push_power, s = attraction * (p_a + p_b) / 2
(p = pull_potential(grid)), R = joint_rest(volume, joints, joint_rest). The push falls
and the squeeze rises as t grows, so there is one crossing in (0, R]; it is found by
BISECTIONS halvings of t / R between MIN_SHARE and 1 (to about 1e-15 of R). R when s
is 0 (and MIN_SHARE * R when the push is 0 and s is not). Power -1 gives the straight spring's k R / (k + s). Reads only the Grid.
new_engine() starts every joint here at rate 0; it swings from there as the energy
around it moves.
"""
from __future__ import annotations

import numpy as np

from physics.cells.engine_settings import EngineSettings
from physics.cells.grid import Grid
from physics.cells.joint_push import MIN_SHARE, joint_push
from physics.cells.joint_rest import joint_rest
from physics.cells.pull_potential import pull_potential

BISECTIONS = 50


def joint_balance(grid: Grid, settings: EngineSettings) -> np.ndarray:
    """Per joint, the thickness in (0, rest] where push and squeeze cancel."""
    exerted = settings.attraction * pull_potential(grid) / 2
    squeeze = exerted[grid.joints[:, 0]] + exerted[grid.joints[:, 1]]
    ones = np.ones(len(squeeze))
    low, high = np.full(len(squeeze), MIN_SHARE), ones.copy()
    for _ in range(BISECTIONS):
        middle = (low + high) / 2
        pushed = joint_push(middle, ones, settings.joint_push, settings.push_power)
        pushed_back = pushed >= squeeze * middle
        low, high = np.where(pushed_back, middle, low), np.where(pushed_back, high, middle)
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    return rest * (low + high) / 2
