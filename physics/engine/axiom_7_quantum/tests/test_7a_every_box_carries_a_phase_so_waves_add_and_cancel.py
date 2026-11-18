"""Clause 7a, "each box's fluid also carries a phase, so waves add and cancel".

The phase is one array over every box of every cell; two waves that meet take one
`pair_rotation` (inside a cell, across a joint), and what comes out depends on their
phases: a quarter turn apart, one side cancels to nothing and the other adds to all.
Axiom 7's own adding and cancelling, the wave a settling draws in, is checked in
`logic/tests/` (equal and opposite waves cancel and the taker keeps its phase).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import pair_rotation

wp.config.quiet = True


@wp.kernel
def _meet(cases: wp.array2d(dtype=wp.float64), after: wp.array2d(dtype=wp.float64)):
    case = wp.tid()
    amount_a, phase_a, amount_b, phase_b = pair_rotation(cases[case, 0], cases[case, 1], cases[case, 2],
                                                         cases[case, 3], cases[case, 4])
    after[case, 0] = amount_a
    after[case, 1] = amount_b


def _amounts_after(phase_a: float, phase_b: float) -> tuple:
    cases = wp.array(np.array([[100.0, phase_a, 100.0, phase_b, math.pi / 4.0]]), dtype=wp.float64, device="cpu")
    after = wp.zeros((1, 2), dtype=wp.float64, device="cpu")
    wp.launch(_meet, dim=1, inputs=[cases, after], device="cpu")
    return tuple(after.numpy()[0])


def test_two_equal_waves_a_quarter_turn_apart_cancel_on_one_side_and_add_on_the_other():
    ahead = _amounts_after(math.pi / 2.0, 0.0)
    assert ahead[0] == pytest.approx(0.0, abs=1e-9) and ahead[1] == pytest.approx(200.0)
    behind = _amounts_after(-math.pi / 2.0, 0.0)
    assert behind[0] == pytest.approx(200.0) and behind[1] == pytest.approx(0.0, abs=1e-9)


def test_two_equal_waves_in_step_trade_nothing():
    assert _amounts_after(0.7, 0.7) == pytest.approx((100.0, 100.0))
