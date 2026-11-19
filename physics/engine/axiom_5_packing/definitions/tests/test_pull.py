"""The bend: in step with the pull's difference while small, never more than `BEND`, the deeper side back."""
import numpy as np
import warp as wp

from physics.engine.axiom_5_packing.definitions.pull import BEND, bent, differing


@wp.kernel
def _read(pulls: wp.array(dtype=wp.vec2f), found: wp.array(dtype=wp.vec3f)):
    share = differing(pulls[wp.tid()][0], pulls[wp.tid()][1])
    turned = bent(0.5, 0.5, share, wp.float64(0.01))
    found[wp.tid()] = wp.vec3f(float(share), turned[0], turned[1])


def _bends(*pulls: tuple) -> np.ndarray:
    """For each (pull before, pull after): how far it bends, and two phases of 0.5 after a hundredth of a turn."""
    found = wp.zeros(len(pulls), dtype=wp.vec3f, device="cpu")
    wp.launch(_read, dim=len(pulls), device="cpu",
              inputs=[wp.array(pulls, dtype=wp.vec2f, device="cpu"), found])
    return found.numpy()


def test_the_bend_follows_the_difference_while_small_and_never_runs_over():
    none, small, one, back, steep = _bends((3, 3), (3, 3.01), (3, 4), (4, 3), (0, 1000))[:, 0]
    assert none == 0 and np.isclose(small, 0.01, rtol=0.02)  # in step with the difference
    assert np.isclose(one, 0.5) and np.isclose(back, -0.5)   # a calm volume's own pull: half
    assert 0.99 < steep < 1                                  # never more than all of it


def test_the_deeper_side_s_phase_is_turned_back_and_the_other_s_on_by_as_much():
    share, low, high = _bends((3, 4))[0]
    half = 0.5 * BEND * 2 * np.pi * share * 0.01
    assert np.allclose([low - 0.5, high - 0.5], [half, -half], atol=1e-6) and half > 0
    assert np.allclose(_bends((3, 3))[0, 1:], 0.5)  # no difference: nothing is turned
