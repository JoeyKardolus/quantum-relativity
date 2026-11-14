"""Per joint, the joint's own volume as a share of its two energy volumes' mean volume.

Contract: `joint_volume_shares(grid)` returns one value per joint:
grid.joint_volume[k] / ((volume[a] + volume[b]) / 2) for joint k between a
and b, rounded to 4 decimals; a value that is not finite becomes 0.0.
"""
from __future__ import annotations

import numpy as np

from physics.cells import Grid


def joint_volume_shares(grid: Grid) -> list[float]:
    """Each joint's volume over the mean volume of the two energy volumes it separates."""
    mean = (grid.volume[grid.joints[:, 0]] + grid.volume[grid.joints[:, 1]]) / 2
    share = np.asarray(grid.joint_volume, dtype=float) / np.maximum(mean, 1e-300)
    return np.round(np.nan_to_num(share, nan=0.0, posinf=0.0, neginf=0.0), 4).tolist()
