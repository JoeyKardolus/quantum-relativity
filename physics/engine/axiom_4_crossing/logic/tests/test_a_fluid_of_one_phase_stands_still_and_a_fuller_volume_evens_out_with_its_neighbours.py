"""`cross` on a real start: nothing drives the fluid, and what is uneven evens out (energy diffuses)."""
from __future__ import annotations

import numpy as np

from physics import engine
from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_4_crossing.logic.cross import cross
from physics.engine.axiom_4_crossing.logic.tests.conftest import held_by, still, universe

EVEN = {"unevenness": 0.0, "inner_unevenness": 0.0}


def _volumes(state) -> list:
    return [int(cell) for cell in np.flatnonzero(state.host("kind") == engine.ENERGY_VOLUME)]


def test_an_even_universe_of_one_phase_is_left_exactly_as_it_is_at_any_friction():
    for friction in (1e-9, 0.5, 1.0):
        state, settings = universe(friction=friction, **EVEN)
        still(state)
        quanta, phase = state.quanta.numpy().copy(), state.phase.numpy().copy()
        for _ in range(5):
            cross(state, settings)
            state.step += 1
        assert np.array_equal(quanta, state.quanta.numpy()) and np.array_equal(phase, state.phase.numpy())
        assert int(state.host("crossed").sum()) == 0 and int(state.host("heat_quanta").sum()) == 0


def test_a_fuller_volume_in_a_still_fluid_evens_out_with_its_neighbours_through_the_joints():
    state, settings = universe(**EVEN)
    still(state)
    lump = _volumes(state)[0]
    state.set_block("quanta", lump, state.block("quanta", lump) * 2)
    count_cells(state)
    start, others = held_by(state, lump), [held_by(state, cell) for cell in _volumes(state)[1:]]
    is_joint = state.host("kind")[:state.count] == engine.JOINT
    joints = state.quanta.numpy()[:state.count][is_joint].copy()
    total = int(np.abs(state.quanta.numpy()).sum())
    held = [start]
    for _ in range(5):
        cross(state, settings)
        count_cells(state)
        state.step += 1
        held.append(held_by(state, lump))
    assert all(later < earlier for earlier, later in zip(held, held[1:])), held
    assert all(held_by(state, cell) > before for cell, before in zip(_volumes(state)[1:], others))
    assert int(np.abs(state.quanta.numpy()).sum()) == total, "a crossing changes no total"
    assert np.array_equal(joints, state.quanta.numpy()[:state.count][is_joint]), "no box of a joint is written"
