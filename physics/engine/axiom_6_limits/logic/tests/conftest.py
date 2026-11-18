"""Shared by the limits logic tests: a small start lattice on the CPU with set energies, and plain checks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_1_energy.definitions import energy_arrays
from physics.engine.axiom_1_energy.definitions.energy_settings import EnergySettings
from physics.engine.axiom_2_swirls.definitions import swirl_arrays
from physics.engine.axiom_3_structure.definitions import structure_arrays
from physics.engine.axiom_3_structure.definitions.grid import Grid
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT, NO_CELL
from physics.engine.axiom_3_structure.definitions.size_law import size_law_host
from physics.engine.axiom_3_structure.definitions.structure_settings import StructureSettings
from physics.engine.axiom_3_structure.definitions.tests.conftest import CODES, lattice_tables
from physics.engine.axiom_4_crossing.definitions import crossing_arrays
from physics.engine.axiom_4_crossing.definitions.crossing_settings import CrossingSettings
from physics.engine.axiom_5_packing.definitions import packing_arrays
from physics.engine.axiom_5_packing.definitions.packing_settings import PackingSettings
from physics.engine.axiom_6_limits.definitions import limits_arrays
from physics.engine.axiom_6_limits.definitions.limits_settings import LimitsSettings
from physics.engine.axiom_6_limits.logic.start_fresh import start_fresh
from physics.engine.axiom_7_quantum.definitions import phase_arrays
from physics.engine.axiom_7_quantum.definitions.quantum_settings import QuantumSettings
from physics.engine.axiom_3_structure.logic.lay_lattice import lay_lattice

wp.config.quiet = True

_OWNERS = (energy_arrays, swirl_arrays, structure_arrays, crossing_arrays, packing_arrays, limits_arrays,
           phase_arrays)
ARRAYS = {name: spec for owner in _OWNERS for name, spec in owner.ARRAYS.items()}
HOST = {name: default for owner in _OWNERS for name, default in owner.HOST.items()}
VOLUME_BOX = 3000
JOINT_BOX = 100


@dataclass(frozen=True)
class Settings(EnergySettings, StructureSettings, CrossingSettings, PackingSettings, LimitsSettings,
               QuantumSettings):
    """Every axiom's settings in one record, as the engine's own will be."""


def recount(state, settings) -> None:
    """Stand in for `count_cells` and `size_cells`: every cell's quanta from its boxes, its length by the size law."""
    kind = state.kind.numpy()
    quanta = np.abs(state.quanta.numpy().astype(np.int64)).reshape(state.capacity, -1).sum(axis=1)
    length = np.array([size_law_host(int(held), settings.size_constant) if kind[cell] != NO_CELL else 0.0
                       for cell, held in enumerate(quanta)], dtype=np.float32)
    state.cell_quanta.assign(quanta)
    state.length.assign(length)


def small_grid(settings: Settings, side: int = 4, spare: int = 8, device: str = "cpu") -> Grid:
    """A start lattice with 3000 a box in every volume, -100 in every joint, seeded phases, rate 1."""
    kind, ids, ways = lattice_tables(side)
    grid = Grid(side, side ** 3 + spare, len(ways) + spare, ARRAYS, HOST, 1, device)
    kinds = np.zeros(grid.capacity, dtype=np.int32)
    kinds[:side ** 3] = kind
    grid.kind.assign(kinds)
    grid.made_kind[:side ** 3] = kind
    lay_lattice(grid, ids, CODES, ways)
    per_box = np.where(kinds == ENERGY_VOLUME, VOLUME_BOX, np.where(kinds == JOINT, -JOINT_BOX, 0))
    grid.quanta.assign(np.broadcast_to(per_box[:, None, None, None], grid.quanta.shape).astype(np.int32))
    phases = np.random.default_rng(6).uniform(-3.0, 3.0, grid.phase.shape).astype(np.float32)
    grid.phase.assign(phases * (kinds != NO_CELL)[:, None, None, None])
    grid.rate.assign(np.ones(grid.capacity, dtype=np.float32))
    recount(grid, settings)
    start_fresh(grid)
    return grid


@pytest.fixture
def settings() -> Settings:
    """The default settings, nothing touched."""
    return Settings()


@pytest.fixture
def state(settings) -> Grid:
    """The 4-a-side start lattice: 8 energy volumes wrapped in 56 joints, every joint fresh."""
    return small_grid(settings)
