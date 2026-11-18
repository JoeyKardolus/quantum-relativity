"""`turn_splits` on a store filled by hand: many splits with the same halves, so shares can be counted."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from physics.engine.axiom_7_quantum.definitions.quantum_settings import QuantumSettings
from physics.engine.axiom_7_quantum.definitions.split_store import empty_store
from physics.engine.axiom_7_quantum.logic.turn_splits import turn_splits

MANY = 20000
CELLS, QUANTA = (0, 2, 8), (210, 30, 60)


def _state(step: int, born: int = 0, seed: int = 5) -> SimpleNamespace:
    """A stand-in state holding `MANY` splits born at `born`, each 210 / 30 / 60 quanta in cells 0 / 2 / 8."""
    store = empty_store()
    store["source"] = np.arange(MANY, dtype=np.int64)
    store["born"] = np.full(MANY, born, dtype=np.int64)
    store["pointer"] = np.zeros(MANY, dtype=np.int64)
    store["directed"] = np.full(MANY, 300, dtype=np.int64)
    store["reacted"] = np.zeros(MANY, dtype=bool)
    store["part_source"] = np.repeat(store["source"], 3)
    store["part_cell"] = np.tile(np.array(CELLS, dtype=np.int64), MANY)
    store["part_quanta"] = np.tile(np.array(QUANTA, dtype=np.int64), MANY)
    return SimpleNamespace(quantum=store, step=step, seed=seed)


def test_a_split_born_this_step_has_not_turned_and_its_pointer_stays():
    state = _state(step=4, born=4)
    turn_splits(state, QuantumSettings())
    assert not state.quantum["pointer"].any()


def test_at_a_whole_turn_the_pointer_lands_on_a_cell_with_its_share_as_its_chance():
    state = _state(step=1)
    turn_splits(state, QuantumSettings())
    shares = [float(np.mean(state.quantum["pointer"] == cell)) for cell in CELLS]
    assert shares == pytest.approx([0.70, 0.10, 0.20], abs=0.012), shares


def test_the_pointer_jumps_only_when_a_whole_turn_is_passed():
    slow = QuantumSettings(slowest_speed=0.25, fastest_speed=0.25)
    for step in (1, 2, 3):
        state = _state(step=step)
        turn_splits(state, slow)
        assert not state.quantum["pointer"].any(), f"a quarter turn a step: no whole turn by step {step}"
    state = _state(step=4)
    turn_splits(state, slow)
    assert np.mean(state.quantum["pointer"] != 0) == pytest.approx(0.30, abs=0.012)


def test_the_speed_is_set_at_the_split_so_splits_born_at_different_steps_turn_differently():
    half = QuantumSettings(slowest_speed=0.25, fastest_speed=1.0)
    early, late = _state(step=3, born=0), _state(step=3, born=1)
    turn_splits(early, half)
    turn_splits(late, half)
    jumped_early, jumped_late = early.quantum["pointer"] != 0, late.quantum["pointer"] != 0
    assert jumped_early.any() and jumped_late.any() and not np.array_equal(jumped_early, jumped_late)


def test_turning_writes_only_the_pointer_and_keeps_no_speed():
    state = _state(step=1)
    before = {name: column.copy() for name, column in state.quantum.items() if isinstance(column, np.ndarray)}
    turn_splits(state, QuantumSettings())
    for name, column in before.items():
        assert name == "pointer" or np.array_equal(column, state.quantum[name]), name
    assert all(column.dtype in (np.int64, np.bool_) for column in state.quantum.values()
               if isinstance(column, np.ndarray)), "the store holds whole numbers only: no speed, no turn count"
