"""The balancing drift keeps a settled even grid's total space within 1% over 20 steps.

On an even grid every volume holds the same energy and has 4 joints of equal
conductance, so the pull potential is the same everywhere and every joint is
pulled alike. Joints start at their rest volume and first settle to where pull
and push balance, losing about 2% of space on the way (a real transient, not
balanced away); once settled, a drift that undoes the space pull squeezes out
in one step undoes it in every step. Total space counts energy volumes and
joints. Without drift the same 20 settled steps shrink it by more than 1%, so
the check can fail.
"""
from __future__ import annotations

from dataclasses import replace

from physics.cells import CellSettings, balancing_drift, build_grid, step

SETTLE = 30
STEPS = 20


def _run(grid, settings, steps):
    for _ in range(steps):
        grid = step(grid, settings)
    return grid


def test_balancing_drift():
    grid, _ = build_grid(600, seed=3, unevenness=0.0)
    settings = CellSettings.all_on()
    drift = balancing_drift(grid, settings)
    assert drift > 0
    balanced = replace(settings, drift=drift)
    settled = _run(grid, balanced, SETTLE)
    start = settled.total_space()
    kept = _run(settled, balanced, STEPS)
    assert kept.cell_count == grid.cell_count
    assert abs(kept.total_space() / start - 1) < 0.01
    assert _run(settled, settings, STEPS).total_space() / start < 0.99
