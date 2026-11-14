"""step is reproducible: it uses no randomness, so the same grid and settings give identical grids."""
from __future__ import annotations

import numpy as np

from physics.cells import CellSettings, build_grid, step


def _run(steps):
    grid, _ = build_grid(400, seed=5, unevenness=0.1)
    settings = CellSettings.all_on()
    for _ in range(steps):
        grid = step(grid, settings)
    return grid


def test_step_is_reproducible():
    first, second = _run(10), _run(10)
    assert first.cell_count == second.cell_count
    for name in ("joints", "energy", "volume", "joint_volume", "squeeze", "charge",
                 "merged_into"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
