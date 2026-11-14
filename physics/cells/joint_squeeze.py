"""Joint squeeze: how close each joint is to its collapse limit, 0..1, for drawing.

Contract: `joint_squeeze(grid, settings)` returns one value per joint,
clip((1 - J / R) / (1 - push_limit), 0, 1), J the joint's volume and R its rest
volume (joint_rest.py): 0 at or above rest, 1 at the limit J = push_limit * R,
where the joint gives way. No rule reads it.
"""
from __future__ import annotations

import numpy as np

from physics.cells.grid import Grid
from physics.cells.joint_rest import joint_rest
from physics.cells.settings import CellSettings


def joint_squeeze(grid: Grid, settings: CellSettings) -> np.ndarray:
    """Per joint, the share of the way from its rest volume down to its collapse limit."""
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    return np.clip((1 - grid.joint_volume / rest) / (1 - settings.push_limit), 0.0, 1.0)
