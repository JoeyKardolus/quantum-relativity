"""`read_books`: every box summed in whole numbers, the ledgers and kept counts beside them, nothing written."""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.definitions.book_law import book_law
from physics.engine.axiom_1_energy.definitions.energy_settings import EnergySettings
from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_1_energy.logic.read_books import read_books
from physics.engine.axiom_1_energy.logic.start_energy import start_energy
from physics.engine.axiom_1_energy.logic.tests.conftest import all_blocks, inner_step
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT

KEYS = {"plus", "minus", "amount", "made_total", "made", "crossed", "heated", "moved_inside", "cut_moves",
        "taken_joint", "taken_volumes", "short_takes", "short_quanta", "torn", "wrong_boxes", "wrong_cells", "empty_cells", "heat"}


def _staged(grid) -> tuple:
    """Every volume 2 a box and every joint -1 a box: 8 volumes and 56 joints."""
    kind = grid.host("kind")
    for cell in range(grid.count):
        grid.set_block("quanta", cell, np.full((6, 6, 6), 2 if kind[cell] == ENERGY_VOLUME else -1, dtype=np.int32))
    count_cells(grid)
    return int(np.flatnonzero(kind == ENERGY_VOLUME)[1]), int(np.flatnonzero(kind == JOINT)[1])


def test_the_totals_are_exact_python_whole_numbers(grid):
    _staged(grid)
    books = read_books(grid)
    assert set(books) == KEYS and all(type(value) is int for value in books.values())
    assert books["plus"] == 8 * 216 * 2 and books["minus"] == 56 * 216 and books["amount"] == 8 * 432 + 56 * 216
    assert books["wrong_boxes"] == books["wrong_cells"] == books["empty_cells"] == 0


def test_a_total_past_64_bits_of_float_is_still_exact(grid):
    volume, joint = _staged(grid)
    grid.set_block("quanta", volume, np.full((6, 6, 6), 2 ** 31 - 1, dtype=np.int32))
    count_cells(grid)
    assert read_books(grid)["plus"] == 7 * 432 + 216 * (2 ** 31 - 1)


def test_wrong_boxes_wrong_cells_and_empty_cells_are_each_named(grid):
    volume, joint = _staged(grid)
    block = grid.block("quanta", volume)
    block[0, 0, 0], block[0, 0, 1] = -5, -1
    grid.set_block("quanta", volume, block)
    block = grid.block("quanta", joint)
    block[3, 3, 3] = 7
    grid.set_block("quanta", joint, block)
    books = read_books(grid)
    assert books["wrong_boxes"] == 3 and books["wrong_cells"] == 2
    assert books["plus"] == 8 * 432 - 4 + 7 and books["minus"] == 56 * 216 - 1 + 6
    count_cells(grid)
    assert read_books(grid)["wrong_cells"] == 0
    grid.set_block("quanta", joint, np.zeros((6, 6, 6), dtype=np.int32))
    count_cells(grid)
    books = read_books(grid)
    assert books["empty_cells"] == 1 and books["wrong_boxes"] == 2
    grid.set_cell("kind", joint, 0)
    assert read_books(grid)["empty_cells"] == 0
    grid.set_block("quanta", joint, np.full((6, 6, 6), -1, dtype=np.int32))
    assert read_books(grid)["wrong_boxes"] == 2 + 216


def test_the_ledgers_and_the_kept_counts_are_read_as_they_stand(grid):
    volume, joint = _staged(grid)
    grid.set_cell("moved_inside", volume, 11)
    grid.set_cell("moved_inside", joint, 4)
    grid.set_cell("cut_moves", volume, 2)
    grid.set_cell("made", joint, 9)
    grid.set_cell("crossed", joint, 13)
    grid.set_cell("heated", joint, 5)
    grid.set_cell("heat_quanta", volume, 40)
    grid.made_total, grid.taken_joint, grid.taken_volumes, grid.short_takes, grid.torn = 21, 22, 23, 24, 25
    grid.short_quanta = 26
    books = read_books(grid)
    assert (books["moved_inside"], books["cut_moves"], books["made"]) == (15, 2, 9)
    assert (books["crossed"], books["heated"], books["heat"]) == (13, 5, 40)
    assert (books["made_total"], books["taken_joint"], books["taken_volumes"], books["short_takes"],
            books["torn"], books["short_quanta"]) == (21, 22, 23, 24, 25, 26)


def test_reading_writes_nothing_and_the_books_stay_exact_while_the_fluid_moves(grid):
    settings = EnergySettings()
    start_energy(grid, settings)
    rng = np.random.default_rng(2)
    for cell in range(grid.count):
        grid.set_block("phase", cell, rng.uniform(-3.0, 3.0, (6, 6, 6)).astype(np.float32))
    start = read_books(grid)
    for step in range(5):
        inner_step(grid, settings)
        quanta, held = all_blocks(grid, "quanta"), grid.host("cell_quanta")
        books = read_books(grid)
        assert np.array_equal(all_blocks(grid, "quanta"), quanta) and np.array_equal(grid.host("cell_quanta"), held)
        assert book_law(start, books) == {"amount": 0, "plus": 0, "minus": 0, "taken": 0}
        assert books["wrong_boxes"] == books["wrong_cells"] == books["empty_cells"] == 0
        assert books["moved_inside"] > 0
    assert grid.start_books is None
