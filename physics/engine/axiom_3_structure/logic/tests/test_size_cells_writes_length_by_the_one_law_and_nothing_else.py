"""size_cells: length is the one constant over the cell's quanta, for both kinds, and only length is written."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT, NO_CELL
from physics.engine.axiom_3_structure.definitions.size_law import size_law_host
from physics.engine.axiom_3_structure.definitions.structure_settings import StructureSettings
from physics.engine.axiom_3_structure.logic.size_cells import size_cells
from physics.engine.axiom_3_structure.logic.tests.conftest import changed_names, snapshot

SETTINGS = StructureSettings()


def _give_quanta(grid, values: dict, everyone: int) -> None:
    """Set `cell_quanta` to `everyone` in every cell and to the named values in the named cells."""
    quanta = np.zeros(grid.capacity, dtype=np.int64)
    quanta[:grid.count] = everyone
    for cell, value in values.items():
        quanta[cell] = value
    grid.cell_quanta.assign(quanta)


def test_the_start_numbers_give_a_volume_10_long_and_a_joint_300_long(grid):
    kind = grid.host("kind")
    quanta = np.zeros(grid.capacity, dtype=np.int64)
    quanta[:grid.count] = np.where(kind == ENERGY_VOLUME, 3000 * 216, 100 * 216)
    grid.cell_quanta.assign(quanta)
    size_cells(grid, SETTINGS)
    length = grid.host("length")
    assert (length[kind == ENERGY_VOLUME] == 10.0).all() and (length[kind == JOINT] == 300.0).all()


def test_a_volume_and_a_joint_holding_the_same_energy_are_the_same_size_and_more_energy_is_smaller(grid):
    kind = grid.host("kind")
    volume, joint = int(np.flatnonzero(kind == ENERGY_VOLUME)[1]), int(np.flatnonzero(kind == JOINT)[3])
    _give_quanta(grid, {volume: 12345, joint: 12345, 0: 7}, everyone=1000)
    size_cells(grid, SETTINGS)
    length = grid.host("length")
    assert length[volume] == length[joint] == size_law_host(12345, SETTINGS.size_constant)
    assert length[0] == size_law_host(7, SETTINGS.size_constant) > length[1] == np.float32(6480.0) > length[volume]


def test_an_empty_cell_is_endlessly_long_and_an_id_that_is_no_cell_has_no_length(grid):
    _give_quanta(grid, {5: 0, 9: 4000}, everyone=1000)
    grid.set_cell("kind", 9, NO_CELL)
    size_cells(grid, SETTINGS)
    length = grid.host("length")
    assert np.isinf(length[5]) and length[9] == 0.0


def test_only_length_is_written_and_only_for_ids_handed_out(grid):
    _give_quanta(grid, {}, everyone=1000)
    grid.cell_quanta.assign(np.full(grid.capacity, 1000, dtype=np.int64))
    before = snapshot(grid)
    size_cells(grid, SETTINGS)
    after = snapshot(grid)
    assert changed_names(before, after) == {"length"}
    assert not after["length"][grid.count:].any()


def test_the_constant_is_the_settings_and_size_follows_a_change_of_energy(grid):
    _give_quanta(grid, {}, everyone=1000)
    size_cells(grid, SimpleNamespace(size_constant=5000.0))
    assert (grid.host("length") == 5.0).all()
    _give_quanta(grid, {3: 2000}, everyone=1000)
    size_cells(grid, SimpleNamespace(size_constant=5000.0))
    assert grid.host("length")[3] == 2.5
