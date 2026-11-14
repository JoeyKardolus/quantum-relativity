"""Joint balance: the volume each joint settles at, where its push back equals the pull on it.

Contract: `joint_balance(grid, settings)` returns one volume per joint (a, b) of the
grid: the fixed point of move_space()'s joint update with the grid held as it is.
That update keeps J * exp(-s) of a joint under pull s = attraction * (p_a + p_b) / 2
(p = pull_potential(grid)) and swells it back by joint_push * (R - J exp(-s)),
R = joint_rest(volume, joints, joint_rest), so it rests where
  J = k R / (1 - (1 - k) exp(-s)),   k = joint_push,
R when nothing pulls (s = 0; with k = 0 too, any J is a balance and R is returned). Reads only the Grid. new_engine() starts
every joint here, so the first step neither squeezes nor swells a joint by itself.
"""
from __future__ import annotations

import numpy as np

from physics.cells.engine_settings import EngineSettings
from physics.cells.grid import Grid
from physics.cells.joint_rest import joint_rest
from physics.cells.pull_potential import pull_potential


def joint_balance(grid: Grid, settings: EngineSettings) -> np.ndarray:
    """Per joint, the volume where pull and push cancel: k R / (1 - (1 - k) exp(-s))."""
    exerted = settings.attraction * pull_potential(grid) / 2
    pull = exerted[grid.joints[:, 0]] + exerted[grid.joints[:, 1]]
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    held = 1 - (1 - settings.joint_push) * np.exp(-pull)
    return np.divide(settings.joint_push * rest, held, out=rest.copy(), where=held > 0)
