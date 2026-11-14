"""build_grid gives about the asked number of energy volumes, each with 4 joints, no coordinates.

A periodic tetrahedralisation has no boundary, so every tetrahedron shares each
of its 4 sides with another one: before any collapse every energy volume has
exactly 4 joints and k = 2n joints. Every joint starts at its rest volume and
every charge at 0. The Grid holds only the contract's fields; the display
positions come back separately.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from physics.cells import CellSettings, Grid, build_grid, joint_rest, rest_volume


@pytest.mark.parametrize("cells", [200, 2500])
def test_build_grid(cells):
    grid, positions = build_grid(cells, seed=4, unevenness=0.1)
    assert abs(grid.cell_count - cells) <= 0.05 * cells
    joints_per_cell = np.bincount(grid.joints.ravel(), minlength=grid.cell_count)
    assert (joints_per_cell == 4).all()
    assert len(grid.joints) == 2 * grid.cell_count
    assert (grid.joints[:, 0] < grid.joints[:, 1]).all()
    assert {field.name for field in dataclasses.fields(Grid)} == {
        "cell_count", "joints", "energy", "volume", "joint_volume", "squeeze", "charge",
        "merged_into"}
    assert positions.shape == (grid.cell_count, 3)
    assert np.allclose(grid.volume, rest_volume(grid.energy))
    assert np.allclose(grid.joint_volume,
                       joint_rest(grid.volume, grid.joints, CellSettings.all_on().joint_rest))
    assert (grid.joint_volume > 0).all() and (grid.charge == 0).all()
    assert np.abs(grid.energy - 1).max() <= 0.1
    assert grid.merged_into is None
