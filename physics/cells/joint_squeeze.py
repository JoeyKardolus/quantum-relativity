"""Joint squeeze: how hard each joint pushes, 0 at rest .. 1 at its collapse limit.

Contract: `joint_squeeze(grid, settings)` returns one value per joint,
clip((1 - t / R) / (1 - push_limit), 0, 1), t the joint's thickness and R its rest
thickness (joint_rest.py): 0 at or above rest, 1 at the limit t = push_limit * R,
where the joint gives way. For drawing (frame_arrays.py); no rule reads it.
"""
from __future__ import annotations

import numpy as np

from physics.cells.engine_settings import EngineSettings
from physics.cells.grid import Grid
from physics.cells.joint_rest import joint_rest


def joint_squeeze(grid: Grid, settings: EngineSettings) -> np.ndarray:
    """Per joint, the share of the way from its rest thickness down to its collapse limit."""
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    return np.clip((1 - grid.thickness / rest) / (1 - settings.push_limit), 0.0, 1.0)
