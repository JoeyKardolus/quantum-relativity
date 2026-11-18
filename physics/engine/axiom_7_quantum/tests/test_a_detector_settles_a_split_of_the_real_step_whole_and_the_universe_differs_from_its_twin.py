"""The real step on a small universe with next to no friction: splits stay open until a detector reacts."""
from __future__ import annotations

import numpy as np

from physics.engine import Settings, new_universe, react, read_books, step
from physics.engine.axiom_7_quantum.definitions.split_store import MOMENTUM, REACTION, amount_of

QUIET = {"friction": 1e-9, "push_strength": 40.0, "tear_length": 200.0}
STEPS = 3


def _universe():
    """A 4-a-side universe run three steps on the CPU; with friction this small no split is reacted with."""
    settings = Settings(**QUIET)
    state = new_universe(settings, cells_per_side=4, seed=1, device="cpu")
    for _ in range(STEPS):
        step(state, settings)
    return state, settings


def test_with_next_to_no_friction_splits_stay_open_and_keep_being_fed():
    state, settings = _universe()
    store = state.quantum
    assert len(store["source"]) == 8 and store["total"]["settled_reaction"] == 0
    assert store["total"]["joined"] > 0 and int(store["born"].max()) == 0


def test_a_detector_in_one_half_s_cell_settles_that_split_whole_in_its_pointer_s_cell():
    state, settings = _universe()
    twin, twin_settings = _universe()
    store = state.quantum
    source = int(store["source"][0])
    cell = int(store["part_cell"][store["part_source"] == source][0])
    plus_before = read_books(state)["plus"]
    react(state, cell)
    step(state, settings)
    step(twin, twin_settings)
    settled = store["settled"]
    mine = settled["source"] == source
    assert mine.any() and settled["cause"][mine].tolist() == [REACTION] and settled["short"][mine].tolist() == [0]
    assert int(settled["amount"][mine][0]) > 0 and read_books(state)["plus"] == plus_before
    assert not np.array_equal(state.host("cell_quanta"), twin.host("cell_quanta")), \
        "the settling moved quanta: the universe is no longer its twin's"
    assert source not in store["source"].tolist() or int(store["born"][store["source"] == source][0]) == STEPS
    assert int(amount_of(twin.quantum).sum()) > 0


def test_energy_that_lost_all_its_momentum_at_its_crossing_settles_whole_in_the_same_step():
    settings = Settings(**{**QUIET, "friction": 1.0})
    state = new_universe(settings, cells_per_side=4, seed=1, device="cpu")
    plus_before = read_books(state)["plus"]
    step(state, settings)
    store = state.quantum
    assert store["step"]["settled_momentum"] == 8 and len(store["source"]) == 0
    assert set(store["settled"]["cause"].tolist()) == {MOMENTUM} and not store["settled"]["short"].any()
    assert store["step"]["drawn_quanta"] > 0 and read_books(state)["plus"] == plus_before
