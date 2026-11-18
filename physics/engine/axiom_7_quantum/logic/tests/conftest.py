"""Shared by axiom 7's logic tests: a 4-a-side start lattice on the CPU with fluid put in by hand."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_1_energy.definitions.energy_arrays import ARRAYS as ENERGY_ARRAYS
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME
from physics.engine.axiom_3_structure.definitions.tests.conftest import start_grid_by_hand
from physics.engine.axiom_4_crossing.definitions.crossing_arrays import ARRAYS as CROSSING_ARRAYS
from physics.engine.axiom_7_quantum.definitions.phase_arrays import ARRAYS as PHASE_ARRAYS
from physics.engine.axiom_7_quantum.definitions.quantum_settings import QuantumSettings
from physics.engine.axiom_7_quantum.logic.open_splits import open_splits
from physics.engine.axiom_7_quantum.logic.read_sent import read_sent
from physics.engine.axiom_7_quantum.logic.start_store import start_store

wp.config.quiet = True

PER_BOX = 10
HELD = PER_BOX * 216
WAY_LEDGERS = ("way_out_a", "way_out_b", "way_lost_a", "way_lost_b", "way_moved")


def set_cells(grid, name: str, values: dict) -> None:
    """Write some rows of a "cell" or "way" array, leaving the others."""
    whole = getattr(grid, name).numpy()
    for row, value in values.items():
        whole[row] = value
    getattr(grid, name).assign(whole)


def way_between(grid, giver: int, taker: int) -> tuple:
    """The slot of the first living way joining two volumes, and the sign of a move from giver to taker."""
    found = [(slot, 1 if way_a == giver else -1)
             for slot, (way_a, way_b) in enumerate(zip(grid.host("way_a").tolist(), grid.host("way_b").tolist()))
             if {way_a, way_b} == {giver, taker}]
    return found[0]


def send(grid, giver: int, taker: int, moved: int, lost: int = 0) -> None:
    """Stage what `cross` would have written for one way: `moved` quanta from giver to taker, `lost` undirected."""
    slot, sign = way_between(grid, giver, taker)
    end = "a" if sign > 0 else "b"
    set_cells(grid, "way_out_" + end, {slot: moved})
    set_cells(grid, "way_lost_" + end, {slot: lost})


def open_fork(grid) -> None:
    """Open, as new moving energy, the splits of the crossing staged with `send`; then clear the staging."""
    rows = read_sent(grid)
    open_splits(grid, rows, np.zeros(len(rows["key"]), dtype=np.int64))
    for name in WAY_LEDGERS:
        getattr(grid, name).zero_()


def next_step(grid) -> None:
    """A new step begins: the ways' ledgers are cleared and the counter goes on."""
    for name in WAY_LEDGERS:
        getattr(grid, name).zero_()
    grid.step += 1


def all_arrays(grid) -> dict:
    """A host copy of every array of the grid, to check what a function left alone."""
    return {name: getattr(grid, name).numpy().copy() for name in grid.arrays}


def held_by(grid, cell: int) -> int:
    """The quanta in one cell's boxes."""
    return int(grid.block("quanta", cell).sum())


@pytest.fixture
def grid():
    """8 energy volumes of 10 plus quanta a box in 56 joints; an empty store; nothing changed this step."""
    made = start_grid_by_hand(extra_arrays={**ENERGY_ARRAYS, **CROSSING_ARRAYS, **PHASE_ARRAYS})
    kind = made.host("kind")
    rng = np.random.default_rng(7)
    for cell in np.flatnonzero(kind == ENERGY_VOLUME):
        made.set_block("quanta", int(cell), np.full((6, 6, 6), PER_BOX))
        made.set_block("phase", int(cell), rng.uniform(-3.0, 3.0, (6, 6, 6)))
    set_cells(made, "cell_quanta", {int(cell): HELD for cell in np.flatnonzero(kind == ENERGY_VOLUME)})
    start_store(made)
    made.changed = []
    return made


@pytest.fixture
def volumes(grid) -> list:
    """The ids of the 8 energy volumes."""
    return [int(cell) for cell in np.flatnonzero(grid.host("kind") == ENERGY_VOLUME)]


@pytest.fixture
def settings() -> QuantumSettings:
    """Axiom 7's own settings; its logic reads no other."""
    return QuantumSettings()
