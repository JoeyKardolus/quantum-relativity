"""Axiom 4's laws on plain numbers: what loses its direction, what is heat, how what arrives is added."""
from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_4_crossing.definitions.arrival_law import arrival_law
from physics.engine.axiom_4_crossing.definitions.crossing_clock import crossing_clock
from physics.engine.axiom_4_crossing.definitions.friction_law import friction_law, heat_leaving, heat_made

wp.config.quiet = True


@wp.kernel
def _laws(numbers: wp.array(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    out[0] = wp.float64(friction_law(wp.int64(numbers[0]), numbers[1], numbers[2]))
    out[1] = wp.float64(heat_leaving(wp.int64(numbers[3]), wp.int64(numbers[4]), wp.int64(numbers[5])))
    out[2] = wp.float64(heat_made(wp.int64(numbers[3]), wp.int64(numbers[4]), wp.int64(numbers[0])))
    out[3] = arrival_law(numbers[6], numbers[7], numbers[8], numbers[9], numbers[10])
    out[4] = crossing_clock(numbers[11], wp.float64(1.0), wp.float64(0.5), wp.float64(0.25))


def _run(crossing=100, friction=0.5, uniform=0.3, moved=100, heat=30, cell=300, holds=900.0, standing=0.2,
         turned=0.4, taken=50.0, lost=0.0, step_time=0.25) -> np.ndarray:
    numbers = wp.array([crossing, friction, uniform, moved, heat, cell, holds, standing, turned, taken, lost,
                        step_time], dtype=wp.float64, device="cpu")
    out = wp.zeros(5, dtype=wp.float64, device="cpu")
    wp.launch(_laws, dim=1, inputs=[numbers], outputs=[out], device="cpu")
    return out.numpy()


def test_friction_takes_its_share_of_what_crosses_in_whole_quanta_and_never_more_than_crosses():
    assert _run(crossing=100, friction=0.5, uniform=0.3)[0] == 50
    assert _run(crossing=101, friction=0.5, uniform=0.3)[0] == 50 and _run(crossing=101, friction=0.5, uniform=0.6)[0] == 51
    assert _run(crossing=7, friction=1.0, uniform=0.99)[0] == 7 and _run(crossing=0, friction=1.0, uniform=0.99)[0] == 0


def test_heat_leaves_a_cell_by_its_share_and_only_what_was_not_heat_is_new_heat():
    assert _run(moved=100, heat=30, cell=300)[1] == 10
    assert _run(moved=100, heat=0, cell=300)[1] == 0 and _run(moved=100, heat=30, cell=0)[1] == 0
    assert _run(crossing=50, moved=100, heat=30)[2] == 35, "of 50 lost, the 70 in 100 that were not heat before"


def test_what_arrives_with_its_direction_goes_on_as_the_step_left_it():
    assert _run(standing=0.2, turned=0.4, lost=0.0)[3] == pytest.approx(0.6)


def test_what_lost_its_direction_arrives_at_rest_so_more_friction_turns_the_box_less():
    turns = [_run(holds=10.0, standing=0.2, turned=0.4, taken=50.0, lost=lost)[3] - 0.2 for lost in (0.0, 25.0, 50.0)]
    assert turns[0] > turns[1] > turns[2] > 0.0
    expected = math.atan2(10 * math.sin(0.6) + 50 * math.sin(0.2), 10 * math.cos(0.6) + 50 * math.cos(0.2))
    assert _run(holds=10.0, standing=0.2, turned=0.4, taken=50.0, lost=50.0)[3] == pytest.approx(expected)
    assert _run(holds=0.0, standing=0.2, turned=0.4, taken=50.0, lost=50.0)[3] == pytest.approx(0.2)


def test_opposite_fluids_of_equal_quanta_cancel_and_the_box_keeps_its_standing_phase():
    assert _run(holds=30.0, standing=0.5, turned=math.pi, taken=20.0, lost=50.0)[3] == pytest.approx(0.5)


def test_a_crossing_runs_on_the_slowest_of_its_three_clocks():
    assert _run(step_time=0.25)[4] == pytest.approx(0.25 * 0.25)
