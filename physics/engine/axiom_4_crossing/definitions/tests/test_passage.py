"""The passage is read off the joint: the rubbing share, the turn it leaves, the step it takes."""
import numpy as np
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import TURN
from physics.engine.axiom_4_crossing.definitions.passage import (
    less_step, passing_turn, row_minus, rub_share)


@wp.kernel
def _read(box_quanta: wp.array2d(dtype=wp.int32), low_phase: wp.float32, high_phase: wp.float32,
          found: wp.array(dtype=wp.float64)):
    minus = row_minus(box_quanta, 0, 1, 2)  # the row of boxes 1 and 3 in a block 2 across
    rub = rub_share(minus, 1000, 3000)
    step = less_step(low_phase, high_phase, rub)
    found[0] = minus
    found[1] = rub
    found[2] = passing_turn(rub)
    found[3] = wp.float64(step[0])
    found[4] = wp.float64(step[1])


def _passage(joint: list, low_phase: float = 0.0, high_phase: float = 0.0) -> np.ndarray:
    found = wp.zeros(5, dtype=wp.float64, device="cpu")
    wp.launch(_read, dim=1, device="cpu", inputs=[
        wp.array([joint], dtype=wp.int32, device="cpu"), low_phase, high_phase, found])
    return found.numpy()


def test_the_rubbing_share_is_the_minus_share_of_the_energy_in_the_passage():
    minus, rub, turn = _passage([0, -1500, 0, -2500, 0, 0, 0, 0])[:3]
    assert minus == 2000 and rub == 0.5 and turn == TURN * 0.5  # 2,000 minus beside 2,000 plus
    assert _passage([0, -6000, 0, -6000, 0, 0, 0, 0])[1] == 0.75  # a fuller joint rubs more
    assert _passage([0] * 8)[0] == 0  # no minus: no joint to pass


def test_friction_takes_its_share_of_the_step_in_phase_the_short_way_round():
    joint = [0, -1500, 0, -2500, 0, 0, 0, 0]  # rubs a half
    assert np.allclose(_passage(joint, 0.8, 0.0)[3:], [0.6, 0.2])
    low, high = _passage(joint, 3.0, -3.0)[3:]  # 0.28 apart across the turn of the angle
    assert np.isclose(low, 3.0 + 0.0708, atol=1e-3) and np.isclose(high, -3.0 - 0.0708, atol=1e-3)
    assert np.allclose(_passage(joint, 1.0, 1.0)[3:], [1.0, 1.0])  # no step: nothing to take
