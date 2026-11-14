"""Pull squeezes joints, push swells them back, and a joint at rest is a thin wall.

  rest     a new engine (500 volumes, seed 8, unevenness 0.1, settled) has its
           joints at about 3% of their two energy volumes' mean volume (default
           joint_rest 0.03; rule 8: 2 to 5%): median ratio within 0.025..0.035.
  pull     with joint_push 0, one move_space shrinks every joint and leaves the
           energy volumes' sizes and energies exactly as the leak and rule 4 give
           (leak 0: unchanged).
  push     with attraction 0 (nothing pulls, nothing leaks), every joint squeezed
           below rest swells each move without passing rest (one above rest thins
           toward it), and all reach rest within 1%, while every
           energy and energy volume stays exactly as it was: a push moves no energy.
"""
from __future__ import annotations

import numpy as np

from physics.cells import EngineSettings, joint_rest, move_space, new_engine


def test_joints_rest_near_three_percent():
    grid = new_engine(500, seed=8, unevenness=0.1).grid
    ratio = grid.joint_volume / joint_rest(grid.volume, grid.joints, 1.0)
    assert 0.025 <= np.median(ratio) <= 0.035


def test_pull_squeezes_and_push_swells_back_without_moving_energy():
    grid = new_engine(500, seed=8, unevenness=0.1).grid
    pulled = move_space(grid, EngineSettings.all_on(joint_push=0.0, leak=0.0))
    assert (pulled.joint_volume < grid.joint_volume).all()
    assert np.array_equal(pulled.energy, grid.energy)
    assert np.array_equal(pulled.volume, grid.volume)
    for _ in range(10):
        pulled = move_space(pulled, EngineSettings.all_on(joint_push=0.0, leak=0.0))
    settings = EngineSettings.all_on(attraction=0.0)
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    assert (pulled.joint_volume < rest).mean() > 0.9
    pushed = pulled
    for _ in range(60):
        swollen = move_space(pushed, settings)
        below = pushed.joint_volume < rest
        assert (swollen.joint_volume[below] > pushed.joint_volume[below]).all()
        assert (swollen.joint_volume[below] <= rest[below]).all()
        assert (swollen.joint_volume[~below] <= pushed.joint_volume[~below]).all()
        assert np.array_equal(swollen.energy, grid.energy)
        assert np.array_equal(swollen.volume, grid.volume)
        pushed = swollen
    assert np.allclose(pushed.joint_volume, rest, rtol=0.01)
