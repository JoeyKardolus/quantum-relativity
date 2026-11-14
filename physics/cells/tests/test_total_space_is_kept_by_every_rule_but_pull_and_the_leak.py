"""Total space (energy volumes plus joints) is kept by every rule but pull and the leak.

Joint push only moves space between a joint and its two energy volumes, and the
energy-volume rule hands the space an energy volume gives up or takes to its own
joints, so with attraction 0 (nothing pulls, so nothing leaks either) one move
keeps the total space to rounding. Pull removes exactly the space it squeezes out
of the joints, recomputed here from the pull potential: sum J (1 - exp(-s)). The
leak changes only energy volumes: the joints come out the same with leak 0 or
not, and every volume ends at rest_volume of its leaked energy (the space it
gained or lost is new or gone: drift).
"""
from __future__ import annotations

import numpy as np

from physics.cells import CellSettings, build_grid, move_space, pull_potential, rest_volume


def test_total_space_kept_but_for_pull_and_the_leak():
    grid, _ = build_grid(400, seed=7, unevenness=0.3)
    for _ in range(5):
        grid = move_space(grid, CellSettings.all_on())
    start = grid.total_space()
    still = move_space(grid, CellSettings.all_on(attraction=0.0))
    assert np.isclose(still.total_space(), start, rtol=1e-12)
    assert not np.allclose(still.joint_volume, grid.joint_volume)
    settings = CellSettings.all_on(leak=0.0)
    potential = pull_potential(grid)
    pull = settings.attraction * (potential[grid.joints[:, 0]] + potential[grid.joints[:, 1]]) / 2
    squeezed_out = float((grid.joint_volume * (1 - np.exp(-pull))).sum())
    pulled = move_space(grid, settings)
    assert np.isclose(pulled.total_space(), start - squeezed_out, rtol=1e-12)
    leaked = move_space(grid, CellSettings.all_on())
    assert np.array_equal(leaked.joint_volume, pulled.joint_volume)
    assert np.array_equal(leaked.volume, rest_volume(leaked.energy))
    assert not np.allclose(leaked.energy, pulled.energy)
