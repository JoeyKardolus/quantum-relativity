"""start_grid: the wrapping A B A B lattice, every relation and way a row, checked against plain loops."""
from __future__ import annotations

import numpy as np
import pytest

from physics.engine.axiom_3_structure.definitions.direction_codes import opposite_host
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT
from physics.engine.axiom_3_structure.definitions.structure_arrays import HOST
from physics.engine.axiom_3_structure.definitions.tests.conftest import CODES, lattice_tables
from physics.engine.axiom_3_structure.logic.start_grid import start_grid
from physics.engine.axiom_3_structure.logic.tests.conftest import TEST_ARRAYS, small_grid


@pytest.mark.parametrize("side", [4, 6, 8])
def test_kinds_entries_and_ways_equal_the_lattice_worked_by_plain_loops(side):
    grid = small_grid(side)
    kind, ids, ways = lattice_tables(side)
    assert grid.count == side ** 3
    assert np.array_equal(grid.host("kind"), kind)
    assert np.array_equal(grid.host("made_kind"), kind)
    for cell in range(grid.count):
        assert grid.neighbours(cell) == list(zip(ids[cell].tolist(), CODES))
    table = np.stack([grid.host(name) for name in ("way_a", "way_b", "way_mouth_a", "way_mouth_b", "way_joint")], 1)
    assert sorted(map(tuple, table.tolist())) == sorted(map(tuple, ways.tolist()))
    assert not grid.host("way_pass").any() and grid.way_passes == 1


def test_an_eighth_of_the_cells_are_volumes_and_none_touches_another(grid):
    kind = grid.host("kind")
    assert np.count_nonzero(kind == ENERGY_VOLUME) == 8 and np.count_nonzero(kind == JOINT) == 56
    for volume in np.flatnonzero(kind == ENERGY_VOLUME):
        entries = grid.neighbours(int(volume))
        assert len(entries) == 26 and all(kind[other] == JOINT for other, code in entries)


def test_one_way_through_a_face_joint_two_through_an_edge_four_through_a_corner():
    grid = small_grid(6)
    kind = grid.host("kind")
    by_volumes = {6: 0, 12: 0, 20: 0}
    for joint in np.flatnonzero(kind == JOINT):
        volumes = sum(1 for other, code in grid.neighbours(int(joint)) if kind[other] == ENERGY_VOLUME)
        ways = grid.ways_of(int(joint))
        assert (volumes, len(ways)) in ((2, 1), (4, 2), (8, 4))
        by_volumes[{1: 6, 2: 12, 4: 20}[len(ways)]] += 1
        for slot, way_a, way_b, mouth_a, mouth_b in ways:
            assert way_a < way_b and mouth_b == opposite_host(mouth_a)
            assert (int(joint), mouth_a) in grid.neighbours(way_a) and (int(joint), mouth_b) in grid.neighbours(way_b)
    assert by_volumes == {6: 81, 12: 81, 20: 27}
    assert grid.way_count == 13 * 27


def test_room_is_the_spare_share_more_and_nothing_else_is_written():
    grid = start_grid(4, 7, 0.5, TEST_ARRAYS, HOST, "cpu")
    assert (grid.capacity, grid.way_capacity, grid.seed, grid.step) == (96, 156, 7, 0)
    assert not grid.host("length").any() and not grid.host("cell_quanta").any()
    assert not grid.quanta.numpy().any() and not grid.host("way_turn").any()
    assert (grid.fell_into == -1).all() and (grid.torn_from == -1).all() and grid.version == 0


def test_the_seed_changes_no_relation():
    one, other = start_grid(4, 1, 0.25, TEST_ARRAYS, HOST, "cpu"), start_grid(4, 99, 0.25, TEST_ARRAYS, HOST, "cpu")
    assert all(one.neighbours(cell) == other.neighbours(cell) for cell in range(one.count))


@pytest.mark.parametrize("side, spare", [(5, 0.25), (2, 0.25), (4, -0.1)])
def test_an_odd_or_tiny_lattice_and_negative_room_are_refused(side, spare):
    with pytest.raises(ValueError):
        start_grid(side, 1, spare, TEST_ARRAYS, HOST, "cpu")
