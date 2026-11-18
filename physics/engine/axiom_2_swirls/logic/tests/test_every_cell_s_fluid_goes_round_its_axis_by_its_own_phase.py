"""Start swirls on a real small universe, and the going round that follows from the fluid alone."""
from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from physics import engine
from physics.engine.axiom_1_energy.logic.flow_inside import flow_inside
from physics.engine.axiom_2_swirls.definitions.swirl_phase import swirl_phase_host
from physics.engine.axiom_2_swirls.tests.check_step import check_step

wp.config.quiet = True

STEPS = 40


def _universe(seed: int = 1):
    settings = engine.Settings()
    return engine.new_universe(settings, cells_per_side=4, seed=seed, device="cpu"), settings


def _flow_only(state, settings) -> None:
    """One step of the fluid inside every cell and nothing else: no crossing, no pull, no event."""
    for axis in range(3):
        for parity in range(2):
            flow_inside(state, settings, axis, parity)
    state.step += 1


def test_every_cell_of_the_start_gets_an_axis_a_sense_and_the_phase_of_one_whole_swirl():
    state, settings = _universe()
    axis, sense = state.host("swirl_axis"), state.host("swirl_sense")
    assert set(axis.tolist()) == {0, 1, 2} and set(sense.tolist()) == {1, -1}
    for cell in (0, 1, 17, 63):
        phase = state.block("phase", cell)
        for box in ((0, 0, 0), (1, 4, 2), (5, 3, 0), (2, 2, 3)):
            assert phase[box] == pytest.approx(swirl_phase_host(axis[cell], sense[cell], *box), abs=1e-15)


def test_two_seeds_give_different_swirls_and_one_seed_the_same():
    first, second, third = _universe(1)[0], _universe(1)[0], _universe(2)[0]
    assert np.array_equal(first.host("swirl_axis"), second.host("swirl_axis"))
    assert np.array_equal(first.host("swirl_sense"), second.host("swirl_sense"))
    assert not np.array_equal(first.host("swirl_sense"), third.host("swirl_sense"))


def test_the_fluid_alone_keeps_every_cell_going_round_its_own_way_and_no_quantum_leaves_its_cell():
    state, settings = _universe()
    held = state.host("cell_quanta").copy()
    for _ in range(STEPS):
        state.moved_inside.zero_()
        _flow_only(state, settings)
        counts = check_step(state, settings)
        assert counts["against_volumes"] == 0 and counts["against_joints"] == 0
        assert counts["unwound_volumes"] == 0 and counts["unwound_joints"] == 0
        assert int(state.host("moved_inside").min()) > 0
    assert np.array_equal(np.abs(state.quanta.numpy()[:state.count]).sum(axis=(1, 2, 3)), held)


def test_a_cell_made_with_the_other_sense_goes_round_the_other_way_so_the_sense_is_no_label():
    state, settings = _universe()
    cell = 0
    sense, axis = int(state.host("swirl_sense")[cell]), int(state.host("swirl_axis")[cell])
    other = np.array([[[swirl_phase_host(axis, -sense, x, y, z) for z in range(engine.BLOCK)]
                       for y in range(engine.BLOCK)] for x in range(engine.BLOCK)])
    state.set_block("phase", cell, other)
    for _ in range(5):
        _flow_only(state, settings)
    assert check_step(state, settings)["against_volumes"] == 1
    state.set_cell("swirl_sense", cell, -sense)
    assert check_step(state, settings)["against_volumes"] == 0
