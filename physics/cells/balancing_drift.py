"""Balancing drift: the drift that keeps the total space of even space steady.

Contract: `balancing_drift(grid, settings)` evens the grid out (every energy
volume holding the mean energy, volume by the energy-volume rule, every joint at
its rest volume; same joints) and returns the drift d that keeps its total space
S = sum V + sum J steady once its joints have settled. Only pull and drift change
S (move_space.py): one step gives S_new = (S - pulled_out) (1 + d), so measuring
r = S_new / S with d = 0 gives d = 1 / r - 1. Joints first settle from rest to
where pull and push balance (a few steps), so SETTLE rounds each measure r on the
current state and then advance it one step with that d; the last d is returned.
Like Einstein's cosmological constant the balance is unstable: a slightly denser
region pulls harder than drift undoes. settings.drift is ignored. Raises
ValueError if a joint collapses on the even grid (no drift can keep it even).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.cells.grid import Grid
from physics.cells.joint_rest import joint_rest
from physics.cells.rest_volume import rest_volume
from physics.cells.settings import CellSettings
from physics.cells.step import step

SETTLE = 30


def balancing_drift(grid: Grid, settings: CellSettings) -> float:
    """Drift that keeps a settled even grid's total space unchanged from step to step."""
    energy = np.full(grid.cell_count, grid.energy.mean())
    volume = rest_volume(energy)
    even = Grid(cell_count=grid.cell_count, joints=grid.joints, energy=energy, volume=volume,
                joint_volume=joint_rest(volume, grid.joints, settings.joint_rest),
                squeeze=np.zeros(len(grid.joints)), charge=np.zeros(grid.cell_count))
    drift = 0.0
    for _ in range(SETTLE):
        measured = _stepped(even, replace(settings, drift=0.0))
        drift = even.total_space() / measured.total_space() - 1.0
        even = _stepped(even, replace(settings, drift=drift))
    return drift


def _stepped(grid: Grid, settings: CellSettings) -> Grid:
    stepped = step(grid, settings)
    if stepped.cell_count != grid.cell_count:
        raise ValueError("joints collapsed on even space: no drift can keep it even")
    return stepped
