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


def _turned(state, cell: int) -> int:
    """Whether a start cell is an odd volume along the first coordinate across the grid's swirl axis."""
    axis = int(state.host("swirl_axis")[cell])
    place = (cell // state.side ** 2, (cell // state.side) % state.side, cell % state.side)
    return (place[(1, 0, 0)[axis]] // 2) % 2


def test_every_cell_of_the_start_gets_the_grid_s_axis_a_sense_and_the_phase_of_one_whole_swirl():
    state, settings = _universe()
    axis, sense = state.host("swirl_axis"), state.host("swirl_sense")
    assert len(set(axis.tolist())) == 1 and set(axis.tolist()) <= {0, 1, 2} and set(sense.tolist()) == {1, -1}
    assert abs(int(sense.sum())) == 0, "half the cells go round each way"
    for cell in (0, 1, 17, 63):
        phase = state.block("phase", cell)
        for box in ((0, 0, 0), (1, 4, 2), (5, 3, 0), (2, 2, 3)):
            wanted = swirl_phase_host(axis[cell], sense[cell], _turned(state, cell), *box)
            assert abs(np.exp(1j * phase[box]) - np.exp(1j * wanted)) < 1e-12


def test_a_cell_holds_what_it_held_dealt_as_a_swirl_holds_it_and_its_core_stands_open():
    state, settings = _universe()
    axis = int(state.host("swirl_axis")[0])
    held = np.abs(state.block("quanta", 0)).astype(np.int64)
    assert held.sum() == state.host("cell_quanta")[0]
    core = np.take(np.take(held, [2, 3], axis=(1, 0, 0)[axis]), [2, 3], axis=(2, 2, 1)[axis]).mean()
    assert core < 0.2 * held.mean() and held.min() >= 1


def test_the_boxes_of_two_volumes_that_face_each_other_across_a_joint_start_as_equal_waves():
    settings = engine.Settings(unevenness=0.0, inner_unevenness=0.0)
    state = engine.new_universe(settings, cells_per_side=4, seed=1, device="cpu")
    quanta, phase = state.quanta.numpy(), state.phase.numpy()
    wave = np.sqrt(np.abs(quanta)) * np.exp(1j * phase)
    ways = [way for joint in np.flatnonzero(state.host("kind") == engine.JOINT).tolist() for way in state.ways_of(joint)]
    assert len(ways) == 8 * 13
    for slot, way_a, way_b, mouth_a, mouth_b in ways:
        step = [mouth_a // 9 - 1, (mouth_a // 3) % 3 - 1, mouth_a % 3 - 1]
        for free in ((1, 1), (2, 4), (4, 3)):
            spare = iter(free)
            box_a = tuple(5 if one > 0 else 0 if one < 0 else next(spare) for one in step)
            box_b = tuple(0 if one > 0 else 5 if one < 0 else box_a[index] for index, one in enumerate(step))
            assert abs(wave[way_a][box_a] - wave[way_b][box_b]) < 0.02 * abs(wave[way_a][box_a])


def test_two_seeds_give_different_swirls_and_one_seed_the_same():
    first, second = _universe(1)[0], _universe(1)[0]
    assert np.array_equal(first.host("swirl_axis"), second.host("swirl_axis"))
    assert np.array_equal(first.host("swirl_sense"), second.host("swirl_sense"))
    others = [_universe(seed)[0] for seed in range(2, 8)]
    assert any(int(other.host("swirl_axis")[0]) != int(first.host("swirl_axis")[0]) for other in others)
    assert any(int(other.host("swirl_sense")[0]) != int(first.host("swirl_sense")[0]) for other in others)


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
    other = np.array([[[swirl_phase_host(axis, -sense, 0, x, y, z) for z in range(engine.BLOCK)]
                       for y in range(engine.BLOCK)] for x in range(engine.BLOCK)])
    state.set_block("phase", cell, other)
    for _ in range(5):
        _flow_only(state, settings)
    assert check_step(state, settings)["against_volumes"] == 1
    state.set_cell("swirl_sense", cell, -sense)
    assert check_step(state, settings)["against_volumes"] == 0
