"""The leak goes to the neighbours in proportion to their pull, through joints only, and stops at the seal.

Hand-built boxes at rest, clocks 1: a centre box of A-energy 16 with four face
neighbours of A 4, 2, 1, 1 (pulls), every one holding B at rest, and no B anywhere
else, so each neighbour's only joint is the centre. One leak of dt: the centre gives
leak * 16 * dt split 4/8, 2/8, 1/8, 1/8 and each neighbour gives back leak * a * dt, so
each neighbour's net gain is leak * dt * a, proportional to its pull (4 : 2 : 1 : 1);
total A exact, B untouched. With the centre's B past the seal point nothing moves;
a neighbour with A but no B takes nothing.
"""
import numpy as np
import pytest

from physics.fluid import QUANTUM, Grid, Settings
from physics.fluid.leak import leak

_NEIGHBOURS = {(3, 2, 2): 4.0, (1, 2, 2): 2.0, (2, 3, 2): 1.0, (2, 1, 2): 1.0}


def _grid(centre_b: float, unjointed: bool) -> Grid:
    settings = Settings.all_on()
    a = np.zeros((5, 5, 5))
    b = np.zeros((5, 5, 5))
    a[2, 2, 2], b[2, 2, 2] = 16.0, centre_b
    for place, pull in _NEIGHBOURS.items():
        a[place], b[place] = pull, settings.b_rest
    if unjointed:
        a[2, 2, 3] = 8.0
    return Grid.from_host(a, b, device="cpu")


def test_net_leak_is_split_by_pull_exact_and_through_joints_only():
    settings = Settings.all_on()
    dt = 0.1
    grid = _grid(settings.b_rest, unjointed=True)
    before_a, before_b = grid.a_energy.numpy().copy(), grid.b_energy.numpy().copy()
    leak(grid, settings, dt)
    gained = (grid.a_energy.numpy() - before_a) * QUANTUM
    for place, pull in _NEIGHBOURS.items():
        assert gained[place] == pytest.approx(settings.leak * dt * pull, rel=1e-6)
    assert gained[2, 2, 3] == 0.0
    assert int(grid.a_energy.numpy().sum()) == int(before_a.sum())
    assert np.array_equal(grid.b_energy.numpy(), before_b)


def test_a_box_past_the_seal_point_leaks_nothing():
    settings = Settings.all_on()
    grid = _grid(1.01 * settings.seal_compression * settings.b_rest, unjointed=False)
    before = grid.a_energy.numpy().copy()
    leak(grid, settings, 0.1)
    assert np.array_equal(grid.a_energy.numpy(), before)
