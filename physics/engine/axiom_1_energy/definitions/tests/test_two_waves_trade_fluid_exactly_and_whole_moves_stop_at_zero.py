"""Pair rotation and whole move, each against hand-worked values."""
from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import pair_rotation
from physics.engine.axiom_1_energy.definitions.whole_move import whole_move

wp.config.quiet = True


@wp.kernel
def _rotate(cases: wp.array2d(dtype=wp.float64), after: wp.array2d(dtype=wp.float64)):
    case = wp.tid()
    amount_a, phase_a, amount_b, phase_b = pair_rotation(cases[case, 0], cases[case, 1], cases[case, 2],
                                                         cases[case, 3], cases[case, 4])
    after[case, 0] = amount_a
    after[case, 1] = phase_a
    after[case, 2] = amount_b
    after[case, 3] = phase_b


@wp.kernel
def _move(flux: wp.array(dtype=wp.float64), uniform: wp.array(dtype=wp.float64), giver: wp.array(dtype=wp.int64),
          taker: wp.array(dtype=wp.int64), moved: wp.array(dtype=wp.int64)):
    case = wp.tid()
    moved[case] = whole_move(flux[case], uniform[case], giver[case], taker[case])


def _rotated(cases: list) -> np.ndarray:
    after = wp.zeros((len(cases), 4), dtype=wp.float64, device="cpu")
    wp.launch(_rotate, dim=len(cases), inputs=[wp.array(np.array(cases), dtype=wp.float64, device="cpu"), after],
              device="cpu")
    return after.numpy()


def test_hand_worked_rotations():
    quarter = math.pi / 2.0
    after = _rotated([
        [100.0, 0.3, 0.0, 0.0, quarter],        # a full handover: all of a is in b, with the phase a had
        [100.0, 0.4, 100.0, 0.4, 0.7],          # equal waves in step: nothing moves and no phase turns
        [100.0, quarter, 100.0, 0.0, math.pi / 4.0],  # a ahead by a quarter turn: all flows a to b
        [100.0, 0.0, 100.0, quarter, math.pi / 4.0],  # b ahead: all flows b to a
        [50.0, 1.0, 20.0, -2.0, 0.0],           # no turn: nothing changes
    ])
    assert after[0, 0] == pytest.approx(0.0, abs=1e-12) and after[0, 2] == pytest.approx(100.0)
    assert after[0, 3] == pytest.approx(0.3)
    assert after[1] == pytest.approx([100.0, 0.4, 100.0, 0.4], abs=1e-12)
    assert after[2, [0, 2]] == pytest.approx([0.0, 200.0], abs=1e-9)
    assert after[3, [0, 2]] == pytest.approx([200.0, 0.0], abs=1e-9)
    assert after[4] == pytest.approx([50.0, 1.0, 20.0, -2.0])


def test_a_rotation_keeps_the_pair_s_amount_and_phases_stay_wrapped():
    rng = np.random.default_rng(3)
    cases = np.column_stack([rng.integers(0, 5000, 200), rng.uniform(-math.pi, math.pi, 200),
                             rng.integers(0, 5000, 200), rng.uniform(-math.pi, math.pi, 200),
                             rng.uniform(0.0, 0.25, 200)]).astype(np.float64)
    after = _rotated(cases.tolist())
    assert after[:, 0] + after[:, 2] == pytest.approx(cases[:, 0] + cases[:, 2], rel=1e-12)
    assert (after[:, [0, 2]] >= 0.0).all() and (np.abs(after[:, [1, 3]]) <= math.pi).all()


def test_hand_worked_whole_moves():
    flux = np.array([2.3, 2.3, -2.3, -2.3, 9.9, -9.9, 0.0, 5.0], dtype=np.float64)
    uniform = np.array([0.69, 0.71, 0.29, 0.31, 0.5, 0.5, 0.999, 0.0], dtype=np.float64)
    giver = np.array([10, 10, 10, 10, 4, 10, 0, 0], dtype=np.int64)
    taker = np.array([10, 10, 10, 10, 10, 3, 0, 7], dtype=np.int64)
    moved = wp.zeros(8, dtype=wp.int64, device="cpu")
    wp.launch(_move, dim=8, inputs=[wp.array(flux, dtype=wp.float64, device="cpu"),
                                    wp.array(uniform, dtype=wp.float64, device="cpu"),
                                    wp.array(giver, dtype=wp.int64, device="cpu"),
                                    wp.array(taker, dtype=wp.int64, device="cpu"), moved], device="cpu")
    # floor of 2.99, 3.01, -2.01, -1.99; cut at the giver's 4; cut at the taker's 3; 0; an empty giver
    assert moved.numpy().tolist() == [2, 3, -3, -2, 4, -3, 0, 0]


def test_whole_moves_are_right_on_average():
    uniform = (np.arange(1000) + 0.5) / 1000.0
    moved = wp.zeros(1000, dtype=wp.int64, device="cpu")
    wp.launch(_move, dim=1000, inputs=[wp.array(np.full(1000, 0.3), dtype=wp.float64, device="cpu"),
                                       wp.array(uniform, dtype=wp.float64, device="cpu"),
                                       wp.array(np.full(1000, 5), dtype=wp.int64, device="cpu"),
                                       wp.array(np.full(1000, 5), dtype=wp.int64, device="cpu"), moved], device="cpu")
    assert moved.numpy().sum() == 300
