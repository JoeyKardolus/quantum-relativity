"""`count_cells`: a cell's quanta is the sum of its boxes ignoring sign, exactly, and nothing else is written."""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_1_energy.logic.tests.conftest import all_blocks
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT


def test_each_cell_counts_its_own_boxes_and_no_other(grid):
    kind = grid.host("kind")
    volume, joint = int(np.flatnonzero(kind == ENERGY_VOLUME)[2]), int(np.flatnonzero(kind == JOINT)[7])
    plus = np.arange(216, dtype=np.int32).reshape(6, 6, 6)
    minus = np.zeros((6, 6, 6), dtype=np.int32)
    minus[1, 2, 3], minus[5, 5, 5] = -40, -2
    grid.set_block("quanta", volume, plus)
    grid.set_block("quanta", joint, minus)
    grid.cell_quanta.fill_(99)
    before = all_blocks(grid, "quanta")
    count_cells(grid)
    held = grid.host("cell_quanta")
    assert held[volume] == 215 * 216 // 2 and held[joint] == 42
    assert held.sum() == held[volume] + held[joint]
    assert np.array_equal(all_blocks(grid, "quanta"), before)


def test_a_count_too_big_for_32_bits_is_still_exact(grid):
    grid.set_block("quanta", 0, np.full((6, 6, 6), 2 ** 31 - 1, dtype=np.int32))
    count_cells(grid)
    assert int(grid.host("cell_quanta")[0]) == 216 * (2 ** 31 - 1)
