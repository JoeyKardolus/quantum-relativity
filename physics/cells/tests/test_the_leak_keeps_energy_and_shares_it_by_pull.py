"""The leak keeps energy exactly and shares each volume's leak over all its neighbours by pull.

share_a->b = pull_b / sum of pull over a's neighbours: every neighbour that
pulls gets a positive share, one pulling twice as hard gets twice as much (not
winner-takes-all), and on an even grid where every volume pulls alike what each
volume leaks out comes back in from its neighbours, so nothing changes.
"""
from __future__ import annotations

import numpy as np

from physics.cells import CellSettings, build_grid, leak, move_space

RATE = 0.1


def test_leak_keeps_energy_on_an_uneven_grid():
    grid, _ = build_grid(400, seed=2, unevenness=0.5)
    pull = np.random.default_rng(2).uniform(0.1, 3.0, grid.cell_count)
    leaked = leak(grid.joints, grid.energy, pull, RATE)
    assert np.isclose(leaked.sum(), grid.energy.sum(), rtol=1e-14)
    assert not np.allclose(leaked, grid.energy)


def test_every_neighbour_gets_a_share_and_twice_the_pull_gets_twice_the_share():
    joints = np.array([[0, 1], [0, 2], [0, 3]])
    alone = np.array([4.0, 0.0, 0.0, 0.0])
    pull = np.array([1.0, 1.0, 2.0, 1.0])
    leaked = leak(joints, alone, pull, RATE)
    received = leaked[1:]
    assert (received > 0).all()
    assert np.isclose(received[1], 2 * received[0])
    assert np.isclose(received[2], received[0])
    assert np.isclose(leaked[0], 4.0 * (1 - RATE))
    assert np.allclose(received, 4.0 * RATE * np.array([1, 2, 1]) / 4)


def test_an_even_grid_with_equal_pulls_does_not_change():
    grid, _ = build_grid(600, seed=3, unevenness=0.0)
    leaked = leak(grid.joints, grid.energy, np.full(grid.cell_count, 0.7), RATE)
    assert np.allclose(leaked, grid.energy, rtol=1e-13)
    with_leak = move_space(grid, CellSettings.all_on(leak=RATE))
    without = move_space(grid, CellSettings.all_on(leak=0.0))
    assert np.allclose(with_leak.energy, without.energy, rtol=1e-12)
