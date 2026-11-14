"""The leak flows only along open seams, keeps energy exactly and shares by pull.

  exact     on an uneven grid with every seam at a random openness the total energy
            is what it was (to rounding) and the energies did move.
  pull      on four volumes around one another (a tetrahedron's worth of joints,
            four seams) with every seam open, volume 0 gives rate * E and each
            neighbour gets pull_b / sum pull: twice the pull, twice the share.
  seams     closing the two seams joint (0, 2) meets cuts volume 2 off: it gets
            nothing from 0, the others still do, and 0 gives only its seams' mean
            openness (1/3) of rate * E.
  steady    with every seam open and equal pulls, energy in proportion to each
            volume's number of joints is what each leaks and gets back.
"""
from __future__ import annotations

import numpy as np

from physics.cells import leak, new_engine

RATE = 0.1
TETRA_JOINTS = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
TETRA_SEAMS = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


def test_leak_keeps_energy_exactly_on_an_uneven_grid():
    grid = new_engine(400, seed=2, unevenness=0.5).grid
    rng = np.random.default_rng(2)
    pull = rng.uniform(0.1, 3.0, grid.cell_count)
    openness = rng.uniform(0.0, 1.0, len(grid.seams))
    leaked = leak(grid.joints, grid.seams, grid.energy, pull, openness, RATE)
    assert np.isclose(leaked.sum(), grid.energy.sum(), rtol=1e-14, atol=0)
    assert not np.allclose(leaked, grid.energy)


def test_every_open_neighbour_gets_a_share_and_twice_the_pull_gets_twice_the_share():
    alone = np.array([4.0, 0.0, 0.0, 0.0])
    pull = np.array([1.0, 1.0, 2.0, 1.0])
    leaked = leak(TETRA_JOINTS, TETRA_SEAMS, alone, pull, np.ones(4), RATE)
    received = leaked[1:]
    assert np.isclose(leaked[0], 4.0 * (1 - RATE))
    assert np.allclose(received, 4.0 * RATE * np.array([1, 2, 1]) / 4)


def test_a_neighbour_behind_closed_seams_gets_nothing():
    alone = np.array([4.0, 0.0, 0.0, 0.0])
    pull = np.array([1.0, 1.0, 2.0, 1.0])
    openness = np.array([0.0, 1.0, 0.0, 1.0])
    leaked = leak(TETRA_JOINTS, TETRA_SEAMS, alone, pull, openness, RATE)
    assert leaked[2] == 0.0
    assert leaked[1] > 0 and np.isclose(leaked[1], leaked[3])
    assert np.isclose(leaked[0], 4.0 * (1 - RATE / 3))
    assert np.isclose(leaked.sum(), 4.0, rtol=1e-15)


def test_with_equal_pulls_energy_in_proportion_to_joints_does_not_change():
    grid = new_engine(600, seed=3, unevenness=0.0).grid
    degree = np.bincount(grid.joints.ravel(), minlength=grid.cell_count).astype(float)
    leaked = leak(grid.joints, grid.seams, degree, np.full(grid.cell_count, 0.7),
                  np.ones(len(grid.seams)), RATE)
    assert np.allclose(leaked, degree, rtol=1e-12)
    assert degree.min() < degree.max()
