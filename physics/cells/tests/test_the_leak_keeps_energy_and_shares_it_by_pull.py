"""The leak keeps energy exactly and shares each volume's leak over all its neighbours by pull.

share_a->b = pull_b / sum of pull over a's neighbours: every neighbour that
pulls gets a positive share, one pulling twice as hard gets twice as much (not
winner-takes-all), and where every volume pulls alike the leak is a random walk
on the neighbour graph: energy in proportion to each volume's number of joints
is what each volume leaks out and gets back, so it does not change (an even
energy does change, since the tessellation's cells have different numbers of
neighbours).
"""
from __future__ import annotations

import numpy as np

from physics.cells import new_engine, leak

RATE = 0.1


def test_leak_keeps_energy_on_an_uneven_grid():
    grid = new_engine(400, seed=2, unevenness=0.5).grid
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


def test_with_equal_pulls_energy_in_proportion_to_joints_does_not_change():
    grid = new_engine(600, seed=3, unevenness=0.0).grid
    degree = np.bincount(grid.joints.ravel(), minlength=grid.cell_count).astype(float)
    leaked = leak(grid.joints, degree, np.full(grid.cell_count, 0.7), RATE)
    assert np.allclose(leaked, degree, rtol=1e-12)
    assert degree.min() < degree.max()
