"""Seam openness: how wide each seam is, 1 relaxed, 0 sealed (bible rule 6).

Contract: `seam_openness(grid, settings)` returns one value in [0, 1] per seam
of grid.seams. A joint's squeeze depth is d = max(0, 1 - J / R), J its volume
and R its rest volume (joint_rest.py): 0 at or above rest, 1 - push_limit at the
push limit, where it collapses. A seam is as wide as its most squeezed joint lets
it be:
  openness = clip(1 - max(d_ab, d_ac, d_bc) / seal, 0, 1)
  seal     = min(settings.seal_point, 1 - push_limit)
a straight line from 1 at rest down to 0 at the seal point; at and past it the
seam is closed and passes nothing. The seal is never deeper than the push
limit, so a joint always seals its seams no later than it collapses. With seal
0 a seam is open only while all its joints are at or above rest.
Raises ValueError for a negative seal_point.
"""
from __future__ import annotations

import numpy as np

from physics.cells.engine_settings import EngineSettings
from physics.cells.grid import Grid
from physics.cells.joint_rest import joint_rest
from physics.cells.seam_joints import seam_joints


def seam_openness(grid: Grid, settings: EngineSettings) -> np.ndarray:
    """Per seam, 1 - (its most squeezed joint's depth / seal point), clipped to [0, 1]."""
    if settings.seal_point < 0:
        raise ValueError("seal_point must be >= 0: a squeeze depth")
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    depth = np.maximum(0.0, 1 - grid.joint_volume / rest)
    deepest = depth[seam_joints(grid.joints, grid.seams, grid.cell_count)].max(axis=1,
                                                                             initial=0.0)
    seal = min(settings.seal_point, 1 - settings.push_limit)
    if seal <= 0:
        return (deepest <= 0).astype(float)
    return np.clip(1 - deepest / seal, 0.0, 1.0)
