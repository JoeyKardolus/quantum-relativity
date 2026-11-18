"""read_cells and read_ways: what the viewer gets is the state's own numbers, as copies, living ways only."""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.logic.read_cells import read_cells
from physics.engine.axiom_3_structure.logic.read_ways import read_ways
from physics.engine.axiom_3_structure.logic.tests.conftest import changed_names, small_grid, snapshot, tear_relations
from physics.engine.axiom_3_structure.logic.relink import relink


def test_read_cells_returns_the_seven_per_cell_arrays_for_the_ids_in_use(grid):
    grid.pull.assign(np.arange(grid.capacity, dtype=np.float32) * -0.5)
    grid.rate.assign(np.full(grid.capacity, 0.25, dtype=np.float32))
    grid.cell_quanta.assign(np.arange(grid.capacity, dtype=np.int64) + 10)
    before = snapshot(grid)
    cells = read_cells(grid)
    assert set(cells) == {"kind", "length", "cell_quanta", "pull", "rate", "fell_into", "torn_from"}
    assert all(values.shape == (grid.count,) for values in cells.values())
    assert cells["pull"][7] == -3.5 and cells["rate"][7] == 0.25 and cells["cell_quanta"][7] == 17
    assert np.array_equal(cells["kind"], grid.host("kind")) and (cells["torn_from"] == -1).all()
    cells["kind"][:] = 0
    cells["fell_into"][:] = 5
    assert changed_names(before, snapshot(grid)) == set()


def test_read_cells_follows_new_ids():
    grid = small_grid(4)
    new_joint, new_volume = tear_relations(grid, 0, 2)
    cells = read_cells(grid)
    assert cells["kind"].shape == (66,) and cells["torn_from"][new_joint] == cells["torn_from"][new_volume] == 0


def test_read_ways_gives_slot_a_b_joint_of_every_living_way(grid):
    before = snapshot(grid)
    ways = read_ways(grid)
    assert ways.shape == (104, 4) and ways.dtype == np.int32
    assert ways[:, 0].tolist() == list(range(104))
    for slot, way_a, way_b, joint in ways.tolist():
        assert (slot, way_a, way_b) in [row[:3] for row in grid.ways_of(joint)]
    assert changed_names(before, snapshot(grid)) == set()


def test_read_ways_leaves_out_a_slot_that_holds_no_way(grid):
    joint = int(read_ways(grid)[5, 3])
    dead = [row[0] for row in grid.ways_of(joint)]
    for other, code in grid.neighbours(joint):
        grid.set_neighbours(other, [entry for entry in grid.neighbours(other) if entry[0] != joint])
    grid.set_neighbours(joint, [])
    relink(grid, [joint])
    ways = read_ways(grid)
    assert len(ways) == 104 - len(dead) and not set(dead) & set(ways[:, 0].tolist())
    assert joint not in ways[:, 3]
