"""The passage is read off the joint: the rubbing share."""
import numpy as np
import warp as wp

from physics.engine.axiom_4_crossing.definitions.passage import row_minus, rub_share

AXIS, ROW = 1, 2  # a joint along direction 1; its row 2 along the ring


@wp.kernel
def _read(box_quanta: wp.array2d(dtype=wp.int32), found: wp.array(dtype=wp.float64)):
    minus = row_minus(box_quanta, 0, 2, 6, 1)  # the outside boxes once round the core at row 2
    rub = rub_share(minus, 1000, 3000)
    found[0] = minus
    found[1] = rub


def _passage(round_core: list) -> np.ndarray:
    """What the passage reads off a joint whose six outside boxes round its core hold `round_core`."""
    joint = np.full((3, 2, 6, 6), 77, dtype=np.int32)  # every other box: never read
    joint[AXIS, 1, :, ROW] = round_core
    found = wp.zeros(2, dtype=wp.float64, device="cpu")
    wp.launch(_read, dim=1, device="cpu", inputs=[
        wp.array(joint.reshape(1, 216), dtype=wp.int32, device="cpu"), found])
    return found.numpy()


THIN = [-1500, -2500, -1500, -2500, -1500, -2500]  # 2,000 a box: rubs a half


def test_the_rubbing_share_is_the_minus_share_of_the_energy_in_the_passage():
    minus, rub = _passage(THIN)
    assert minus == 2000 and rub == 0.5  # 2,000 minus beside 2,000 plus
    assert _passage([-6000] * 6)[1] == 0.75  # a fuller joint rubs more
    assert _passage([0] * 6)[0] == 0  # no minus: no joint to pass

