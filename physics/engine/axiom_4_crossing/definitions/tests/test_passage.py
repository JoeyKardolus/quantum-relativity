"""The passage is read off the joint: the rubbing share, the turn it leaves, the step it takes."""
import numpy as np
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import TURN
from physics.engine.axiom_4_crossing.definitions.passage import (
    layer_part, less_step, passing_turn, row_minus, rub_share)

AXIS, ROW = 1, 2  # a joint along direction 1; its row 2 along the ring


@wp.kernel
def _read(box_quanta: wp.array2d(dtype=wp.int32), low_phase: wp.float32, high_phase: wp.float32,
          found: wp.array(dtype=wp.float64)):
    minus = row_minus(box_quanta, 0, 2, 6, 1)  # the outside boxes once round the core at row 2
    rub = rub_share(minus, 1000, 3000)
    step = less_step(low_phase, high_phase, rub)
    found[0] = minus
    found[1] = rub
    found[2] = passing_turn(rub, 6)
    found[3] = wp.float64(step[0])
    found[4] = wp.float64(step[1])


def _passage(round_core: list, low_phase: float = 0.0, high_phase: float = 0.0) -> np.ndarray:
    """What the passage reads off a joint whose six outside boxes round its core hold `round_core`."""
    joint = np.full((3, 2, 6, 6), 77, dtype=np.int32)  # every other box: never read
    joint[AXIS, 1, :, ROW] = round_core
    found = wp.zeros(5, dtype=wp.float64, device="cpu")
    wp.launch(_read, dim=1, device="cpu", inputs=[
        wp.array(joint.reshape(1, 216), dtype=wp.int32, device="cpu"), low_phase, high_phase, found])
    return found.numpy()


THIN = [-1500, -2500, -1500, -2500, -1500, -2500]  # 2,000 a box: rubs a half


def test_the_rubbing_share_is_the_minus_share_of_the_energy_in_the_passage():
    minus, rub, turn = _passage(THIN)[:3]
    assert minus == 2000 and rub == 0.5 and turn == TURN * 0.5  # 2,000 minus beside 2,000 plus
    assert _passage([-6000] * 6)[1] == 0.75  # a fuller joint rubs more
    assert _passage([0] * 6)[0] == 0  # no minus: no joint to pass


def test_a_step_at_the_facing_boxes_counts_for_less_the_more_boxes_a_cell_has():
    assert passing_turn(wp.float64(0.5), 6) == TURN * 0.5  # the least boxes: the whole turn
    assert passing_turn(wp.float64(0.5), 24) == TURN * 0.5 / 4 and layer_part(12) == 0.5


def test_friction_takes_its_share_of_the_step_in_phase_the_short_way_round():
    assert np.allclose(_passage(THIN, 0.8, 0.0)[3:], [0.6, 0.2])
    low, high = _passage(THIN, 3.0, -3.0)[3:]  # 0.28 apart across the turn of the angle
    assert np.isclose(low, 3.0 + 0.0708, atol=1e-3) and np.isclose(high, -3.0 - 0.0708, atol=1e-3)
    assert np.allclose(_passage(THIN, 1.0, 1.0)[3:], [1.0, 1.0])  # no step: nothing to take
