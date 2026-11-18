"""`check_step` itself, on a small real start with no crossing: every count it must return, all exact.

This runs today, on the CPU, with only axiom 1's part of the step; the record tests
beside it need the whole step.
"""
from __future__ import annotations

import numpy as np
import pytest

from physics.engine.axiom_1_energy.definitions.energy_settings import EnergySettings
from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_1_energy.logic.read_books import read_books
from physics.engine.axiom_1_energy.logic.start_energy import start_energy
from physics.engine.axiom_1_energy.logic.tests.conftest import inner_step, small_grid
from physics.engine.axiom_1_energy.tests.check_step import check_step

SETTINGS = EnergySettings()


def _started():
    grid = small_grid()
    start_energy(grid, SETTINGS)
    rng = np.random.default_rng(3)
    for cell in range(grid.count):
        grid.set_block("phase", cell, rng.uniform(-3.0, 3.0, (6, 6, 6)))
    grid.clock.fill_(1.0)
    return grid


def test_it_refuses_a_state_whose_start_was_never_read():
    with pytest.raises(ValueError):
        check_step(_started(), SETTINGS)


def test_a_closed_start_keeps_exact_books_and_moves_in_every_cell_for_50_steps():
    grid = _started()
    grid.start_books = read_books(grid)
    for step in range(50):
        inner_step(grid, SETTINGS)
        counts = check_step(grid, SETTINGS)
        assert all(type(value) in (int, float) for value in counts.values())
        assert (counts["book_amount"], counts["book_plus"], counts["book_minus"], counts["book_taken"]) == (0, 0, 0, 0)
        assert counts["wrong_boxes"] == counts["wrong_cells"] == counts["still_cells"] == 0
        assert counts["moved_inside"] > 0 and counts["amount"] == grid.start_books["amount"]


def test_an_unbooked_change_and_a_booked_one_are_told_apart():
    grid = _started()
    grid.start_books = read_books(grid)
    joint = int(np.flatnonzero(grid.host("kind") == -1)[0])
    block = grid.block("quanta", joint)
    block[0, 0, 0] -= 7
    grid.set_block("quanta", joint, block)
    count_cells(grid)
    counts = check_step(grid, SETTINGS)
    assert (counts["book_amount"], counts["book_plus"], counts["book_minus"]) == (7, 0, 7)
    grid.made_total += 7
    counts = check_step(grid, SETTINGS)
    assert (counts["book_amount"], counts["book_plus"], counts["book_minus"], counts["book_taken"]) == (0, 0, 0, 0)


def test_a_cell_that_holds_energy_and_does_not_move_is_counted_as_standing_still():
    grid = _started()
    grid.start_books = read_books(grid)
    inner_step(grid, SETTINGS)
    grid.clock.fill_(1.0)
    grid.set_cell("moved_inside", 5, 0)
    assert check_step(grid, SETTINGS)["still_cells"] == 1
