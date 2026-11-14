"""Total space (energy volumes plus joints) is kept by every rule but pull and drift.

Joint push only moves space between a joint and its two energy volumes, and the
energy-volume rule and energy flow hand the space an energy volume gives up or
takes to its own joints, so with attraction 0 and drift 0 one move keeps the
total space to rounding. Drift alone multiplies it by (1 + drift). Pull alone
removes exactly the space it squeezes out of the joints, recomputed here from
the pull potential: sum J (1 - exp(-s)).
"""
from __future__ import annotations

import numpy as np

from physics.cells import CellSettings, build_grid, move_space, pull_potential


def test_total_space_kept_but_for_pull_and_drift():
    grid, _ = build_grid(400, seed=7, unevenness=0.3)
    for _ in range(5):
        grid = move_space(grid, CellSettings.all_on())
    start = grid.total_space()
    still = move_space(grid, CellSettings.all_on(attraction=0.0))
    assert np.isclose(still.total_space(), start, rtol=1e-12)
    assert not np.allclose(still.joint_volume, grid.joint_volume)
    drifted = move_space(grid, CellSettings.all_on(attraction=0.0, drift=0.01))
    assert np.isclose(drifted.total_space(), start * 1.01, rtol=1e-12)
    settings = CellSettings.all_on()
    potential = pull_potential(grid)
    pull = settings.attraction * (potential[grid.joints[:, 0]] + potential[grid.joints[:, 1]]) / 2
    squeezed_out = float((grid.joint_volume * (1 - np.exp(-pull))).sum())
    pulled = move_space(grid, settings)
    assert np.isclose(pulled.total_space(), start - squeezed_out, rtol=1e-12)
