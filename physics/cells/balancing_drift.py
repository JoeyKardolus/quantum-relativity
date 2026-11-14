"""Balancing drift: the drift that keeps even space the same size on average.

Contract: `balancing_drift(grid, settings)` evens the grid out (every volume
holding the mean energy, volume by the energy-volume rule; same joints) and
returns the drift d for which one step leaves that even grid's total volume
exactly unchanged. On even space every volume feels the same pull, so step 4
gives V' = V (1 + d) q with one factor q for all, and step 5 gives
V_new = V (V' / V) ** (1/2) = V sqrt((1 + d) q). Measuring
r = sum V_new / sum V with d = 0 gives q = r ** 2, so d = 1 / r ** 2 - 1.
Like Einstein's cosmological constant the balance is unstable: a slightly
denser region pulls harder than drift undoes. settings.drift is ignored.
Raises ValueError if a joint collapses on the even grid (push_limit is below
the background squeeze, and no drift can keep space even).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.cells.grid import Grid
from physics.cells.rest_volume import rest_volume
from physics.cells.settings import CellSettings
from physics.cells.step import step


def balancing_drift(grid: Grid, settings: CellSettings) -> float:
    """Drift that makes one step leave an even grid's total volume unchanged."""
    energy = np.full(grid.cell_count, grid.energy.mean())
    even = Grid(cell_count=grid.cell_count, joints=grid.joints, energy=energy,
                volume=rest_volume(energy), squeeze=np.zeros(len(grid.joints)))
    stepped = step(even, replace(settings, drift=0.0))
    if stepped.cell_count != even.cell_count:
        raise ValueError("joints collapsed on even space: push_limit is below the background squeeze")
    kept = float(stepped.volume.sum() / even.volume.sum())
    return 1.0 / kept ** 2 - 1.0
