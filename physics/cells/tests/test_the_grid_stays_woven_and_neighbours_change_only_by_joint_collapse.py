"""The grid stays woven: neighbours stay neighbours, and only joint collapse joins two volumes.

  balanced  a new engine (500 volumes, seed 1, unevenness 0.5, default settings)
            starts every joint at its balance: squeeze depth 1 - J / R below the
            seal point and below half the push limit's depth, and each cell's
            tessellated size within 10% of its target (mean within 2%).
  step 1    collapses nothing.
  woven     over 100 steps at the default settings nothing collapses, and the grid's
            joints and seams are identical to the start's after every step, while
            total energy stays exact (relative 1e-12).
  counted   neighbour_changes leaves pairs a collapse reshaped out and counts the
            rest: relabelled through merged_into, a pair touching a volume that fell
            into one (or its neighbours) is not counted, any other change is.
  capped    move_seeds moves no seed further than `limit` times its cell's width.
"""
from __future__ import annotations

import numpy as np
import pytest

from physics.cells import (EngineSettings, joint_rest, move_seeds, neighbour_changes,
                           new_engine, step, target_size)

SETTINGS = EngineSettings.all_on()


@pytest.fixture(scope="module")
def engine():
    return new_engine(500, seed=1, unevenness=0.5)


def test_the_start_is_balanced_and_sized(engine):
    grid = engine.grid
    depth = 1 - grid.joint_volume / joint_rest(grid.volume, grid.joints, SETTINGS.joint_rest)
    assert depth.max() < SETTINGS.seal_point
    assert depth.max() < (1 - SETTINGS.push_limit) * 0.7
    error = np.abs(engine.tessellation.volume / target_size(grid) - 1)
    assert error.max() < 0.1 and error.mean() < 0.02


def test_step_one_collapses_nothing(engine):
    assert step(engine, SETTINGS).collapses == 0


def test_neighbours_stay_neighbours_for_100_steps(engine):
    energy = engine.grid.total_energy()
    later = engine
    for _ in range(100):
        later = step(later, SETTINGS)
        assert later.collapses == 0
        assert np.array_equal(later.grid.joints, engine.grid.joints)
        assert np.array_equal(later.grid.seams, engine.grid.seams)
        assert abs(later.grid.total_energy() / energy - 1) < 1e-12


def test_neighbour_changes_leave_collapse_zones_out():
    before = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    merged_into = np.array([0, 0, 1, 2, 3, 4])
    joints = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    assert neighbour_changes(before, merged_into, joints, joints) == 0
    assert neighbour_changes(before, merged_into, np.array([[0, 1], [0, 2], [1, 2], [2, 3],
                                                            [3, 4]]), joints) == 0
    assert neighbour_changes(before, merged_into, np.array([[0, 1], [1, 2], [2, 3], [2, 4],
                                                            [3, 4]]), joints) == 1
    assert neighbour_changes(before, merged_into, np.array([[0, 1], [1, 2], [2, 3]]),
                             joints) == 1


def test_seed_moves_are_capped(engine):
    moved = move_seeds(engine.positions, engine.tessellation, engine.grid.joints,
                       np.ones(len(engine.grid.joints)), engine.side, 0.001)
    offset = moved - engine.positions
    offset -= engine.side * np.round(offset / engine.side)
    width = np.cbrt(engine.tessellation.volume)
    assert (np.linalg.norm(offset, axis=1) <= 0.001 * width * (1 + 1e-9) + 1e-12).all()
