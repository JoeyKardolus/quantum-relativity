"""`fill_joints`: whole quanta, exactly the level, the boxes' proportions kept, only joints, only at the start."""
from __future__ import annotations

import numpy as np
import pytest

from physics.engine.axiom_1_energy.definitions.energy_settings import EnergySettings
from physics.engine.axiom_1_energy.logic.fill_joints import fill_joints
from physics.engine.axiom_1_energy.logic.start_energy import start_energy
from physics.engine.axiom_1_energy.logic.tests.conftest import all_blocks
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT


def test_every_joint_holds_its_level_exactly_no_box_is_empty_and_no_volume_is_touched(grid):
    settings = EnergySettings()
    start_energy(grid, settings)
    kind = grid.host("kind")
    before = all_blocks(grid, "quanta")[:grid.count].copy()
    levels = np.where(kind == JOINT, 50000.4 + np.arange(grid.count), 7.0)
    moved = fill_joints(grid, settings, levels)
    after = all_blocks(grid, "quanta")[:grid.count]
    held = grid.host("cell_quanta")
    assert np.array_equal(held[kind == JOINT], np.rint(levels[kind == JOINT]).astype(np.int64))
    assert (after[kind == JOINT] <= -1).all() and np.array_equal(after[kind == ENERGY_VOLUME], before[kind == ENERGY_VOLUME])
    assert moved == pytest.approx(np.max(np.abs(held[kind == JOINT] - 648000.0 * before[kind == JOINT].reshape(-1, 216).sum(axis=1) / -648000.0)
                                         / -before[kind == JOINT].reshape(-1, 216).sum(axis=1)), rel=1e-6)


def test_a_joint_keeps_the_proportions_of_its_boxes_and_a_level_below_one_a_box_is_one_a_box(grid):
    settings = EnergySettings()
    start_energy(grid, settings)
    joint = int(np.flatnonzero(grid.host("kind") == JOINT)[0])
    shape = np.abs(grid.block("quanta", joint)).astype(np.float64)
    levels = np.full(grid.count, 3.0)
    levels[joint] = 216 + 4 * shape.sum()
    fill_joints(grid, settings, levels)
    filled = np.abs(grid.block("quanta", joint)).astype(np.float64)
    assert np.abs(filled - (1.0 + 4.0 * shape)).max() <= 1.0, "one each, the rest by what each box held"
    other = int(np.flatnonzero(grid.host("kind") == JOINT)[1])
    assert (grid.block("quanta", other) == -1).all() and grid.host("cell_quanta")[other] == 216


def test_only_a_start_is_laid_this_way(grid):
    start_energy(grid, EnergySettings())
    grid.step = 1
    with pytest.raises(ValueError, match="lays a start"):
        fill_joints(grid, EnergySettings(), np.full(grid.count, 1000.0))
