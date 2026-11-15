"""Topology changes only by its two events: a volume split adds one cell, a joint collapse removes one.

A split also gives every neighbour whose face the cut crossed a second joint (one with
each half), and a collapse joins a common neighbour's two joints into one, so the joint
count moves by one plus those."""
from dataclasses import replace

import numpy as np

from physics.cells import (EngineSettings, cell_volumes, engine_stats, joint_rest, new_engine,
                           step)


def _neighbours(joints, cell):
    return set(joints[joints[:, 0] == cell, 1]) | set(joints[joints[:, 1] == cell, 0])


def test_one_thin_cell_splits_into_two_with_a_new_joint_at_rest_between_them():
    engine = new_engine(200, 6, 0.5)
    grid = engine.grid
    density = np.sort(grid.energy / grid.volume)
    settings = EngineSettings.all_on(leak=0.0, push_limit=0.0,
                                     split_threshold=float(density[:2].mean()))
    parent = int(np.argmin(grid.energy / grid.volume))
    after = step(engine, settings)
    assert after.splits == 1 and after.grid.cell_count == grid.cell_count + 1
    child = grid.cell_count
    tear = np.flatnonzero((after.grid.joints == [parent, child]).all(axis=1))
    assert len(tear) == 1
    assert np.isclose(after.grid.energy[[parent, child]].sum(), grid.energy[parent], rtol=1e-14)
    both = _neighbours(after.grid.joints, parent) & _neighbours(after.grid.joints, child)
    assert len(after.grid.joints) == len(grid.joints) + 1 + len(both)
    assert (cell_volumes(after.corners, after.fan) > 0).all()
    assert engine_stats(after)["neighbour_changes"] == 0


def test_one_joint_below_its_limit_collapses_its_two_cells_into_one():
    """The joint is set at a tenth of rest with the push off, so it stays there for the step; every
    other joint sits at its balance and moves a few percent at most."""
    engine = new_engine(200, 6, 0.5)
    grid = engine.grid
    settings = EngineSettings.all_on(leak=0.0, split_threshold=0.0, joint_push=0.0)
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    chosen = int(np.argmax(grid.thickness / rest))
    thickness = grid.thickness.copy()
    thickness[chosen] = 0.1 * rest[chosen]
    rate = grid.rate.copy()
    rate[chosen] = 0.0
    first, second = grid.joints[chosen]
    common = (_neighbours(grid.joints, first) & _neighbours(grid.joints, second))
    after = step(replace(engine, grid=replace(grid, thickness=thickness, rate=rate)), settings)
    assert after.collapses == 1 and after.grid.cell_count == grid.cell_count - 1
    assert len(after.grid.joints) == len(grid.joints) - 1 - len(common)
    assert np.isclose(after.grid.energy.max(), max(after.grid.energy.max(),
                                                   grid.energy[[first, second]].sum()))
    assert np.isclose(after.grid.total_energy(), grid.total_energy(), rtol=1e-14)
