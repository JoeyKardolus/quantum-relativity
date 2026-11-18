"""Shared by the energy logic's tests: a small real grid, the part of a step axiom 1 owns, two meters.

`small_grid` is a 4-a-side start lattice (8 energy volumes wrapped in 56 joints) on the
CPU holding every axiom's arrays, every clock at full rate. Nothing crosses in these
tests (crossing is axiom 4), so every cell here is a closed cell.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_1_energy.logic.clear_ledgers import clear_ledgers
from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_1_energy.logic.flow_inside import flow_inside
from physics.engine.axiom_3_structure.definitions.grid import Grid
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK
from physics.engine.axiom_3_structure.definitions.tests.conftest import CODES, lattice_tables
from physics.engine.axiom_3_structure.logic.lay_lattice import lay_lattice

wp.config.quiet = True

_PACKAGES = (("axiom_1_energy", "energy"), ("axiom_2_swirls", "swirl"), ("axiom_3_structure", "structure"),
             ("axiom_4_crossing", "crossing"), ("axiom_5_packing", "packing"), ("axiom_6_limits", "limits"),
             ("axiom_7_quantum", "phase"))


def all_arrays() -> tuple:
    """Every axiom's ARRAYS and HOST merged; a name in two axioms raises."""
    arrays, host = {}, {}
    for package, name in _PACKAGES:
        module = importlib.import_module(f"physics.engine.{package}.definitions.{name}_arrays")
        for target, source in ((arrays, module.ARRAYS), (host, module.HOST)):
            clash = set(target) & set(source)
            if clash:
                raise ValueError(f"{sorted(clash)} is in two axioms")
            target.update(source)
    return arrays, host


def small_grid(side: int = 4, spare: int = 8, seed: int = 1) -> Grid:
    """The start lattice on the CPU with every axiom's arrays; every clock runs at full rate."""
    kind, ids, ways = lattice_tables(side)
    arrays, host = all_arrays()
    grid = Grid(side, side ** 3 + spare, len(ways) + spare, arrays, host, seed, "cpu")
    values = np.zeros(grid.capacity, dtype=np.int32)
    values[:side ** 3] = kind
    grid.kind.assign(values)
    grid.made_kind[:side ** 3] = kind
    lay_lattice(grid, ids, CODES, ways)
    grid.rate.fill_(1.0)
    return grid


def inner_step(state, settings) -> None:
    """The part of a step that axiom 1 owns, in the step's order, with no crossing."""
    clear_ledgers(state)
    for axis in range(3):
        for parity in range(2):
            flow_inside(state, settings, axis, parity)
    count_cells(state)
    state.step += 1


def all_blocks(state, name: str) -> np.ndarray:
    """A host copy of a whole box array."""
    return getattr(state, name).numpy().copy()


@pytest.fixture
def grid() -> Grid:
    """A fresh small grid for one test."""
    return small_grid()
