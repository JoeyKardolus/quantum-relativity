"""Clauses 3a to 3e on the start: plus never touches plus, minus wraps it in one web, all as relations."""
from __future__ import annotations

import numpy as np
import pytest

from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT
from physics.engine.axiom_3_structure.definitions.structure_arrays import ARRAYS, HOST
from physics.engine.axiom_3_structure.logic.check_pattern import check_pattern
from physics.engine.axiom_3_structure.logic.read_ways import read_ways
from physics.engine.axiom_3_structure.logic.start_grid import start_grid


@pytest.mark.parametrize("side", [4, 16])
def test_a_fresh_grid_holds_the_whole_pattern(side):
    grid = start_grid(side, 1, 0.25, ARRAYS, HOST, "cpu")
    found = check_pattern(grid)
    assert found.pop("web_pieces") == 1 and found.pop("fewest_joints") == 26
    assert not any(found.values()), found
    kind = grid.host("kind")
    volumes = np.count_nonzero(kind == ENERGY_VOLUME)
    assert volumes * 8 == grid.count and np.count_nonzero(kind == JOINT) == grid.count - volumes
    assert len(read_ways(grid)) == 13 * volumes


def test_every_one_of_the_26_joints_round_a_volume_carries_a_way():
    grid = start_grid(16, 1, 0.25, ARRAYS, HOST, "cpu")
    volume = int(np.flatnonzero(grid.host("kind") == ENERGY_VOLUME)[100])
    mouths = set()
    for joint, code in grid.neighbours(volume):
        ways = [row for row in grid.ways_of(joint) if volume in row[1:3]]
        assert len(ways) == 1
        mouths.add(ways[0][3] if ways[0][1] == volume else ways[0][4])
    assert len(mouths) == 26


def test_the_state_gives_a_cell_no_place_only_relations_and_a_length():
    grid = start_grid(4, 1, 0.25, ARRAYS, HOST, "cpu")
    per_cell = {name for name, spec in ARRAYS.items() if spec[1] == "cell"}
    assert per_cell == {"kind", "length", "adj_at", "adj_n"}
    assert not any(spec[1] == "box" for spec in ARRAYS.values())
    assert grid.host("length").shape == (64,)
