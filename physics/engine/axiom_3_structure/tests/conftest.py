"""Shared by axiom 3's tests: every axiom's default settings in one record, and a hand-filled whole state.

The real runs are in `physics/engine/tests/conftest.py`; the run tests import them from there.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import warp as wp

from physics.engine.axiom_1_energy.definitions import energy_arrays, energy_settings
from physics.engine.axiom_2_swirls.definitions import swirl_arrays
from physics.engine.axiom_3_structure.definitions import structure_arrays, structure_settings
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME
from physics.engine.axiom_3_structure.logic.size_cells import size_cells
from physics.engine.axiom_3_structure.logic.start_grid import start_grid
from physics.engine.axiom_4_crossing.definitions import crossing_arrays, crossing_settings
from physics.engine.axiom_5_packing.definitions import packing_arrays, packing_settings
from physics.engine.axiom_6_limits.definitions import limits_arrays, limits_settings
from physics.engine.axiom_7_quantum.definitions import phase_arrays, quantum_settings

wp.config.quiet = True

_ARRAY_FILES = (energy_arrays, swirl_arrays, structure_arrays, crossing_arrays, packing_arrays, limits_arrays,
                phase_arrays)
_SETTINGS = (energy_settings.EnergySettings, structure_settings.StructureSettings,
             crossing_settings.CrossingSettings, packing_settings.PackingSettings, limits_settings.LimitsSettings,
             quantum_settings.QuantumSettings)


def default_settings() -> SimpleNamespace:
    """Every axiom's default settings in one record, as the engine's `Settings` will hold them."""
    return SimpleNamespace(**{name: value for record in _SETTINGS
                              for name, value in dataclasses.asdict(record()).items()})


def filled_state(side: int = 4, seed: int = 1, device: str = "cpu"):
    """A whole state (every axiom's arrays) on the start lattice, its fluid set by hand from the seed.

    Not a universe: volumes get uneven plus quanta and joints minus quanta with any
    phase, every clock runs at full rate and the pull is a small uneven field, so a
    logic function of another axiom can be tried on it.
    """
    arrays = {name: spec for one in _ARRAY_FILES for name, spec in one.ARRAYS.items()}
    host = {name: default for one in _ARRAY_FILES for name, default in one.HOST.items()}
    state = start_grid(side, seed, 0.25, arrays, host, device)
    draws = np.random.default_rng(seed)
    kind = state.host("kind")
    quanta = np.zeros((state.capacity, 6, 6, 6), dtype=np.int32)
    quanta[:state.count] = draws.integers(50, 5000, size=(state.count, 6, 6, 6))
    quanta[:state.count][kind != ENERGY_VOLUME] = -draws.integers(1, 200, size=(np.count_nonzero(
        kind != ENERGY_VOLUME), 6, 6, 6))
    state.quanta.assign(quanta)
    state.phase.assign(draws.uniform(-3.0, 3.0, size=quanta.shape).astype(np.float32))
    state.cell_quanta.assign(np.abs(quanta).sum(axis=(1, 2, 3)).astype(np.int64))
    state.rate.assign(np.ones(state.capacity, dtype=np.float32))
    state.pull.assign(draws.uniform(-1e-3, 0.0, size=state.capacity).astype(np.float32))
    size_cells(state, default_settings())
    return state
