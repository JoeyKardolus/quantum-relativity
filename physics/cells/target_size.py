"""Target size: the volume each cell of the tessellation should take.

Contract: `target_size(grid)` returns per energy volume its own volume plus half
of each of its joints' volumes: the cell drawn around an energy volume holds the
energy volume and its half of every joint on its faces. The targets sum to
grid.total_space(). More energy, smaller volume (rest_volume.py), smaller target.
"""
from __future__ import annotations

import numpy as np

from physics.cells.grid import Grid


def target_size(grid: Grid) -> np.ndarray:
    """Energy volume plus half of every joint around it, per energy volume."""
    half = grid.joint_volume / 2
    count = grid.cell_count
    return (grid.volume + np.bincount(grid.joints[:, 0], half, count)
            + np.bincount(grid.joints[:, 1], half, count))
