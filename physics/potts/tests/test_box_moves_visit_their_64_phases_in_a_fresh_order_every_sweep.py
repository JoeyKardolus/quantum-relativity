"""Box moves visit their 64 phases in a fresh order every sweep, so no axis goes first.

A sweep runs its 64 phases (offsets mod 4 in x, y, z) one after another, and a later
phase sees the moves of an earlier one. In one fixed x-major order (round 3,
2026-11-16) directions the cube treats as the same carried flow differently, x / y / z
0.959 / 0.969 / 0.974 and -x 5% below x, all numerics; shuffled every sweep they agree,
0.966 / 0.968 / 0.966. Two sweeps on a foam of 4 cells of 4 boxes: each launches every
phase once, the two orders differ, and neither is the x-major one.
"""
import importlib

import warp as wp

from physics.potts import Settings, new_universe
from physics.potts.tests._gpu import needs_gpu

moves_module = importlib.import_module("physics.potts.box_moves")


@needs_gpu
def test_two_sweeps_launch_all_64_phases_each_in_different_orders(monkeypatch):
    universe = new_universe(4, 4, seed=3)
    launched = []
    real_launch = wp.launch

    def recording_launch(kernel, *args, **kwargs):
        if kernel is moves_module._moves:
            offset = kwargs["inputs"][16]
            launched.append(int(offset[0]) * 16 + int(offset[1]) * 4 + int(offset[2]))
        return real_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", recording_launch)
    moves_module.box_moves(universe, Settings.all_on())
    moves_module.box_moves(universe, Settings.all_on())
    first, second = launched[:64], launched[64:]
    assert len(launched) == 128
    assert sorted(first) == list(range(64)) and sorted(second) == list(range(64))
    assert first != second
    assert first != list(range(64)) and second != list(range(64))
