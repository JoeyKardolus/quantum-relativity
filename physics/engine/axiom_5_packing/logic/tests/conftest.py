"""Shared by the packing logic's tests: a start lattice on the CPU with numbers set by hand.

A function test, not a run: the lattice is the plain start pattern (energy volumes where
all three indices are even, joints everywhere else, wrapping round), and each test sets
the per-cell numbers its function reads.
"""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_1_energy.definitions.energy_arrays import ARRAYS as ENERGY_ARRAYS
from physics.engine.axiom_1_energy.definitions.energy_settings import EnergySettings
from physics.engine.axiom_3_structure.definitions.grid import Grid
from physics.engine.axiom_3_structure.definitions.kinds import BOXES, ENERGY_VOLUME, JOINT
from physics.engine.axiom_3_structure.definitions.size_law import size_law_host
from physics.engine.axiom_3_structure.definitions.structure_arrays import ARRAYS as STRUCTURE_ARRAYS
from physics.engine.axiom_3_structure.definitions.structure_arrays import HOST as STRUCTURE_HOST
from physics.engine.axiom_3_structure.definitions.structure_settings import StructureSettings
from physics.engine.axiom_3_structure.definitions.tests.conftest import CODES, lattice_tables
from physics.engine.axiom_5_packing.definitions.packing_arrays import ARRAYS as PACKING_ARRAYS
from physics.engine.axiom_5_packing.definitions.packing_settings import PackingSettings
from physics.engine.axiom_3_structure.logic.lay_lattice import lay_lattice

wp.config.quiet = True

ARRAYS = {**STRUCTURE_ARRAYS, **PACKING_ARRAYS, "cell_quanta": ENERGY_ARRAYS["cell_quanta"]}
SPARE = 3


def make_settings(**changes) -> types.SimpleNamespace:
    """The settings the packing logic reads, at their defaults unless changed."""
    values = {"step_time": EnergySettings().step_time, "size_constant": StructureSettings().size_constant,
              **dataclasses.asdict(PackingSettings())}
    values.update(changes)
    return types.SimpleNamespace(**values)


def set_cells(state: Grid, name: str, values) -> None:
    """Write a whole per-cell or per-way array from the values of its rows in use."""
    target = getattr(state, name)
    whole = target.numpy().copy()
    whole[:len(values)] = values
    target.assign(whole)


def make_lattice(side: int) -> Grid:
    """The start pattern on the CPU, `SPARE` ids of room left over, every number 0."""
    kind, ids, ways = lattice_tables(side)
    state = Grid(side, side ** 3 + SPARE, len(ways) + SPARE, ARRAYS, STRUCTURE_HOST, 1, "cpu")
    set_cells(state, "kind", kind)
    state.made_kind[:side ** 3] = kind
    lay_lattice(state, ids, CODES, ways)
    return state


def fill_even(state: Grid, settings, volume_box: int = 3000, joint_box: int = 100) -> None:
    """Give every volume and every joint the same quanta a box, and the length the size law gives."""
    kind = state.host("kind")
    quanta = np.where(kind == ENERGY_VOLUME, volume_box * BOXES, joint_box * BOXES).astype(np.int64)
    set_quanta(state, settings, quanta)


def set_quanta(state: Grid, settings, quanta) -> None:
    """Write `cell_quanta` and the lengths the size law gives for them."""
    set_cells(state, "cell_quanta", np.asarray(quanta, dtype=np.int64))
    set_cells(state, "length", np.array([size_law_host(int(one), settings.size_constant) for one in quanta],
                                        dtype=np.float32))


def snapshot(state: Grid) -> dict:
    """Every array of the state as host copies, capacity and all."""
    return {name: getattr(state, name).numpy().copy() for name in state.arrays}


def changed_names(before: dict, state: Grid) -> set:
    """The arrays that differ from an earlier snapshot."""
    after = snapshot(state)
    return {name for name in before if not np.array_equal(before[name], after[name], equal_nan=True)}


@pytest.fixture
def settings() -> types.SimpleNamespace:
    """The default settings the packing logic reads."""
    return make_settings()


@pytest.fixture
def state() -> Grid:
    """The 4-a-side start lattice (8 energy volumes, 56 joints), every number 0."""
    return make_lattice(4)


__all__ = ["BOXES", "ENERGY_VOLUME", "JOINT", "changed_names", "fill_even", "make_lattice", "make_settings",
           "set_cells", "set_quanta", "snapshot"]
