"""Inside a cell: the fluid moves in whole quanta, keeps every one, and no plus energy pushes plus energy."""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_1_energy.logic.tests.conftest import all_blocks, inner_step, small_grid
from physics.engine.axiom_1_energy.definitions.energy_settings import EnergySettings


def _filled(volume_box: int = 3000) -> tuple:
    grid = small_grid()
    kind = grid.host("kind")
    for cell in range(grid.count):
        grid.set_block("quanta", cell, np.full((6, 6, 6), int(kind[cell]) * volume_box, dtype=np.int32))
    count_cells(grid)
    return grid, int(np.flatnonzero(kind == 1)[0])


def test_an_even_fluid_of_one_phase_stands_as_it_is_so_the_block_s_sides_are_no_force():
    grid, volume = _filled()
    before = all_blocks(grid, "quanta")[:grid.count].copy()
    phases = all_blocks(grid, "phase")[:grid.count].copy()
    for _ in range(40):
        inner_step(grid, EnergySettings())
        assert int(grid.host("moved_inside").sum()) == 0
    assert np.array_equal(all_blocks(grid, "quanta")[:grid.count], before)
    assert np.allclose(all_blocks(grid, "phase")[:grid.count], phases, atol=1e-12)


def test_a_crowded_box_empties_into_its_neighbours_as_a_wave_and_every_quantum_is_kept():
    grid, volume = _filled()
    block = grid.block("quanta", volume)
    block[2, 2, 2] = 6000
    grid.set_block("quanta", volume, block)
    count_cells(grid)
    held = int(grid.host("cell_quanta")[volume])
    inner_step(grid, EnergySettings())
    after = grid.block("quanta", volume)
    assert int(after.sum()) == held and (after >= 0).all()
    assert 0 < 6000 - int(after[2, 2, 2]) < 3000
    for _ in range(200):
        inner_step(grid, EnergySettings())
    assert int(grid.block("quanta", volume).sum()) == held
    assert int(grid.host("moved_inside")[volume]) > 0


def test_a_slow_clock_slows_the_fluid_and_does_not_stop_it():
    moved = {}
    for rate in (1.0, 1e-3, 1e-6):
        grid, volume = _filled()
        block = grid.block("quanta", volume)
        block[2, 2, 2] = 2_000_000_000
        grid.set_block("quanta", volume, block)
        grid.rate.fill_(rate)
        for _ in range(50):
            inner_step(grid, EnergySettings())
        moved[rate] = 2_000_000_000 - int(grid.block("quanta", volume)[2, 2, 2])
    assert moved[1.0] > moved[1e-3] > moved[1e-6] > 0
