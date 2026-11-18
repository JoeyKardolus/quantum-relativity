"""`flow_inside`: two boxes of one cell trade whole quanta by their phases; nothing leaves the block."""
from __future__ import annotations

import math

import numpy as np
import pytest

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform_host
from physics.engine.axiom_1_energy.definitions.energy_settings import EnergySettings
from physics.engine.axiom_1_energy.definitions.passes import FLOW_ROUND
from physics.engine.axiom_1_energy.logic.flow_inside import flow_inside
from physics.engine.axiom_1_energy.logic.tests.conftest import all_blocks
from physics.engine.axiom_3_structure.definitions.kinds import ENERGY_VOLUME, JOINT

SETTINGS = EnergySettings()
QUARTER = math.pi / 2.0


def _two_boxes(grid, cell: int, sign: int, lower: tuple, upper: tuple, ahead: str) -> None:
    """1000 quanta in each of two boxes of one cell, one a quarter turn ahead of the other."""
    quanta = np.zeros((6, 6, 6), dtype=np.int32)
    phase = np.zeros((6, 6, 6), dtype=np.float64)
    quanta[lower] = quanta[upper] = sign * 1000
    phase[lower if ahead == "lower" else upper] = QUARTER
    grid.set_block("quanta", cell, quanta)
    grid.set_block("phase", cell, phase)


def _cell_of(grid, sort: int) -> int:
    return int(np.flatnonzero(grid.host("kind") == sort)[5])


@pytest.mark.parametrize("sort", [ENERGY_VOLUME, JOINT])
@pytest.mark.parametrize("axis, lower, upper", [(0, (2, 1, 4), (3, 1, 4)), (1, (5, 0, 0), (5, 1, 0)),
                                                 (2, (3, 3, 4), (3, 3, 5))])
def test_fluid_flows_from_the_box_whose_phase_is_ahead_by_the_hand_worked_whole_number(grid, sort, axis, lower, upper):
    cell = _cell_of(grid, sort)
    grid.step = 9
    _two_boxes(grid, cell, sort, lower, upper, "lower")
    flow_inside(grid, SETTINGS, axis, 0)
    flow = 1000.0 * math.sin(2.0 * SETTINGS.step_time)
    box = cell * 216 + lower[0] * 36 + lower[1] * 6 + lower[2]
    moved = math.floor(flow + draw_uniform_host(grid.seed, 9, FLOW_ROUND + axis * 2, box))
    after = grid.block("quanta", cell)
    assert moved in (479, 480)
    assert after[lower] == sort * (1000 - moved) and after[upper] == sort * (1000 + moved)
    assert np.abs(after).sum() == 2000 and np.count_nonzero(after) == 2
    assert grid.host("moved_inside")[cell] == moved and grid.host("moved_inside").sum() == moved
    assert not grid.host("cut_moves").any()
    phase = grid.block("phase", cell)
    assert phase[lower] == pytest.approx(QUARTER + SETTINGS.step_time, abs=1e-6)     # what the two differ by turned
    assert phase[upper] == pytest.approx(SETTINGS.step_time, abs=1e-6)


def test_the_flow_turns_round_when_the_other_box_is_ahead(grid):
    cell = _cell_of(grid, ENERGY_VOLUME)
    _two_boxes(grid, cell, 1, (0, 0, 0), (1, 0, 0), "upper")
    flow_inside(grid, SETTINGS, 0, 0)
    after = grid.block("quanta", cell)
    assert after[0, 0, 0] in (1479, 1480) and after[0, 0, 0] + after[1, 0, 0] == 2000


def test_a_slow_clock_moves_less(grid):
    cell = _cell_of(grid, ENERGY_VOLUME)
    grid.set_cell("rate", cell, 0.5)
    _two_boxes(grid, cell, 1, (0, 0, 0), (1, 0, 0), "lower")
    flow_inside(grid, SETTINGS, 0, 0)
    assert int(grid.block("quanta", cell)[1, 0, 0]) - 1000 in (247, 248)


def test_equal_boxes_in_step_trade_nothing_and_a_fuller_box_empties_into_a_thinner_one(grid):
    cell = _cell_of(grid, ENERGY_VOLUME)
    quanta = np.zeros((6, 6, 6), dtype=np.int32)
    quanta[0, 0, 0] = quanta[1, 0, 0] = 700
    quanta[0, 2, 0], quanta[1, 2, 0] = 1000, 0
    grid.set_block("quanta", cell, quanta)
    flow_inside(grid, SETTINGS, 0, 0)
    after = grid.block("quanta", cell)
    assert after[0, 0, 0] == 700 and after[1, 0, 0] == 700
    assert int(after[1, 2, 0]) in (61, 62) and after[0, 2, 0] + after[1, 2, 0] == 1000


def test_a_parity_takes_only_its_own_pairs_and_no_pair_leaves_the_block(grid):
    cell = _cell_of(grid, ENERGY_VOLUME)
    _two_boxes(grid, cell, 1, (1, 0, 0), (2, 0, 0), "lower")
    flow_inside(grid, SETTINGS, 0, 0)
    after = grid.block("quanta", cell)
    assert after[0, 0, 0] + after[1, 0, 0] == 1000 and after[2, 0, 0] + after[3, 0, 0] == 1000
    _two_boxes(grid, cell, 1, (1, 0, 0), (2, 0, 0), "lower")
    flow_inside(grid, SETTINGS, 0, 1)
    assert grid.block("quanta", cell)[2, 0, 0] in (1479, 1480)
    _two_boxes(grid, cell, 1, (5, 0, 0), (0, 0, 0), "lower")
    for parity in (0, 1):
        flow_inside(grid, SETTINGS, 0, parity)
    after = grid.block("quanta", cell)
    assert after[3:, 0, 0].sum() == 1000 and after[:3, 0, 0].sum() == 1000


def test_the_six_calls_take_every_face_inside_a_block_once_and_touch_no_other_cell(grid):
    cell = _cell_of(grid, ENERGY_VOLUME)
    rng = np.random.default_rng(0)
    grid.set_block("quanta", cell, rng.integers(0, 4000, (6, 6, 6)).astype(np.int32))
    grid.set_block("phase", cell, rng.uniform(-3.0, 3.0, (6, 6, 6)).astype(np.float32))
    quanta, phase = all_blocks(grid, "quanta"), all_blocks(grid, "phase")
    for axis in range(3):
        for parity in range(2):
            flow_inside(grid, SETTINGS, axis, parity)
    after, after_phase = all_blocks(grid, "quanta"), all_blocks(grid, "phase")
    others = np.arange(grid.capacity) != cell
    assert np.array_equal(after[others], quanta[others]) and np.array_equal(after_phase[others], phase[others])
    assert after[cell].sum() == quanta[cell].sum() and (after[cell] >= 0).all()
    assert not np.array_equal(after[cell], quanta[cell])
    assert (np.abs(after_phase[cell]) <= np.float32(math.pi)).all()
    assert grid.host("moved_inside")[cell] > 0 and np.count_nonzero(grid.host("moved_inside")) == 1


def test_a_dead_id_and_empty_boxes_do_nothing(grid):
    cell = _cell_of(grid, ENERGY_VOLUME)
    _two_boxes(grid, cell, 1, (0, 0, 0), (1, 0, 0), "lower")
    grid.set_cell("kind", cell, 0)
    before = all_blocks(grid, "phase")
    flow_inside(grid, SETTINGS, 0, 0)
    assert grid.block("quanta", cell)[0, 0, 0] == 1000 and not grid.host("moved_inside").any()
    assert np.array_equal(all_blocks(grid, "phase"), before)


def test_the_same_seed_and_step_give_the_same_flow_and_it_refuses_a_wrong_axis(grid):
    cell = _cell_of(grid, ENERGY_VOLUME)
    results = []
    for step in (3, 3, 4):
        grid.step = step
        rng = np.random.default_rng(1)
        grid.set_block("quanta", cell, rng.integers(0, 50, (6, 6, 6)).astype(np.int32))
        grid.set_block("phase", cell, rng.uniform(-3.0, 3.0, (6, 6, 6)).astype(np.float32))
        flow_inside(grid, SETTINGS, 1, 0)
        results.append(grid.block("quanta", cell))
    assert np.array_equal(results[0], results[1]) and not np.array_equal(results[0], results[2])
    with pytest.raises(ValueError):
        flow_inside(grid, SETTINGS, 3, 0)
