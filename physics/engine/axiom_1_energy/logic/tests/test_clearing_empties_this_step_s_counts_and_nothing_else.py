"""`clear_ledgers`: every per-step array is 0 again, every other array and every kept count is as it was."""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.logic.clear_ledgers import clear_ledgers


def test_only_the_per_step_arrays_and_the_events_are_cleared(grid):
    before = {}
    for name, (dtype, shape, each_step) in grid.arrays.items():
        array = getattr(grid, name)
        array.fill_(3)
        before[name] = array.numpy().copy()
    grid.changed = [{"what": "tear"}]
    grid.made_total, grid.taken_joint, grid.taken_volumes, grid.short_takes, grid.torn = 5, 6, 7, 8, 9
    clear_ledgers(grid)
    cleared = sorted(name for name, spec in grid.arrays.items() if spec[2])
    assert "moved_inside" in cleared and "cut_moves" in cleared and "way_moved" in cleared and "made" in cleared
    for name, (dtype, shape, each_step) in grid.arrays.items():
        after = getattr(grid, name).numpy()
        if each_step:
            assert not after.any(), name
        else:
            assert np.array_equal(after, before[name]), name
    assert grid.changed == []
    assert (grid.made_total, grid.taken_joint, grid.taken_volumes, grid.short_takes, grid.torn) == (5, 6, 7, 8, 9)
