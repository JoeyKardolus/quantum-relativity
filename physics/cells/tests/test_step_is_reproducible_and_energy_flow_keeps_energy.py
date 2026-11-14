"""step is reproducible, and energy flow alone keeps the total energy exactly.

step uses no randomness, so the same grid and settings give identical grids.
Energy flow moves energy through joints, each flux leaving one volume and
entering the other: the total is kept, the spread shrinks.
"""
from __future__ import annotations

import numpy as np

from physics.cells import CellSettings, balancing_drift, build_grid, energy_flow, step


def _run(steps):
    grid, _ = build_grid(400, seed=5, unevenness=0.1)
    settings = CellSettings.all_on()
    settings = CellSettings.all_on(drift=balancing_drift(grid, settings))
    for _ in range(steps):
        grid = step(grid, settings)
    return grid


def test_step_is_reproducible():
    first, second = _run(10), _run(10)
    assert first.cell_count == second.cell_count
    for name in ("joints", "energy", "volume", "squeeze", "merged_into"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    grid, _ = build_grid(400, seed=5, unevenness=0.1)
    flowed = energy_flow(grid, grid.energy, 0.5)
    assert np.isclose(flowed.sum(), grid.energy.sum(), rtol=1e-13)
    assert flowed.std() < grid.energy.std()
