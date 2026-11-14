"""The balancing drift keeps an even grid's total volume within 1% over 20 steps.

On an even grid every volume holds the same energy and has 4 joints of equal
conductance, so the pull potential is the same everywhere and every volume
shrinks by the same factor; a drift that undoes one step undoes every step.
Without drift the same run shrinks by more than 1%, so the check can fail.
"""
from __future__ import annotations

from dataclasses import replace

from physics.cells import CellSettings, balancing_drift, build_grid, step

STEPS = 20


def _run(grid, settings):
    for _ in range(STEPS):
        grid = step(grid, settings)
    return grid


def test_balancing_drift():
    grid, _ = build_grid(600, seed=3, unevenness=0.0)
    settings = CellSettings.all_on()
    drift = balancing_drift(grid, settings)
    assert drift > 0
    start = grid.volume.sum()
    balanced = _run(grid, replace(settings, drift=drift))
    assert balanced.cell_count == grid.cell_count
    assert abs(balanced.volume.sum() / start - 1) < 0.01
    assert _run(grid, settings).volume.sum() / start < 0.99
