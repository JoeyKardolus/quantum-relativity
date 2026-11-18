"""Shared by the structure definitions' tests: a start lattice worked out the slow, plain way."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_3_structure.definitions.grid import Grid
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT
from physics.engine.axiom_3_structure.definitions.structure_arrays import ARRAYS, HOST
from physics.engine.axiom_3_structure.logic.lay_lattice import lay_lattice

wp.config.quiet = True

CODES = [code for code in range(27) if code != 13]


def lattice_tables(side: int) -> tuple:
    """Kinds, the 26 neighbour ids of every cell and every way of a wrapping lattice, by plain loops."""
    count = side ** 3
    kind = np.zeros(count, dtype=np.int32)
    ids = np.zeros((count, 26), dtype=np.int32)
    for cell in range(count):
        place = (cell // (side * side), (cell // side) % side, cell % side)
        kind[cell] = ENERGY_VOLUME if all(part % 2 == 0 for part in place) else JOINT
        for column, code in enumerate(CODES):
            step = (code // 9 - 1, (code // 3) % 3 - 1, code % 3 - 1)
            there = [(place[axis] + step[axis]) % side for axis in range(3)]
            ids[cell, column] = (there[0] * side + there[1]) * side + there[2]
    ways = []
    for joint in np.flatnonzero(kind == JOINT):
        for column, code in enumerate(CODES):
            first, second = int(ids[joint, column]), int(ids[joint, CODES.index(26 - code)])
            if code < 13 and kind[first] == ENERGY_VOLUME and kind[second] == ENERGY_VOLUME and first != second:
                ways.append((first, second, 26 - code, code, joint) if first < second
                            else (second, first, code, 26 - code, joint))
    return kind, ids, np.array(ways, dtype=np.int32)


def start_grid_by_hand(side: int = 4, spare: int = 8, extra_arrays: dict | None = None) -> Grid:
    """A Grid on the CPU holding the start lattice."""
    kind, ids, ways = lattice_tables(side)
    grid = Grid(side, side ** 3 + spare, len(ways) + spare, {**ARRAYS, **(extra_arrays or {})}, HOST, 1, "cpu")
    values = np.zeros(grid.capacity, dtype=np.int32)
    values[:side ** 3] = kind
    grid.kind.assign(values)
    grid.made_kind[:side ** 3] = kind
    lay_lattice(grid, ids, CODES, ways)
    return grid


@pytest.fixture
def grid() -> Grid:
    """The 4-a-side start lattice: 8 energy volumes wrapped in 56 joints."""
    return start_grid_by_hand()
