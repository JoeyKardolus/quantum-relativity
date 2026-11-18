"""The pull grows with both volumes and falls with distance; the push rises steeply; they balance."""
import numpy as np
import warp as wp

from physics.engine.axiom_5_packing.definitions.pull_push import GIVE, given, pull_across, push_back


@wp.kernel
def _read(held: wp.array(dtype=wp.vec3d), found: wp.array(dtype=wp.vec3d)):
    left, right, joint = held[wp.tid()][0], held[wp.tid()][1], held[wp.tid()][2]
    pull, push = pull_across(left, right, joint), push_back(joint)
    found[wp.tid()] = wp.vec3d(pull, push, given(wp.vec2d(pull, push), wp.float64(0.01)))


def _forces(*held: tuple) -> np.ndarray:
    """Pull, push and what is given in a hundredth of a turn, for each (left, right, joint)."""
    found = wp.zeros(len(held), dtype=wp.vec3d, device="cpu")
    wp.launch(_read, dim=len(held), device="cpu",
              inputs=[wp.array(held, dtype=wp.vec3d, device="cpu"), found])
    return found.numpy()


def test_two_calm_volumes_hold_their_joint_at_half_their_size():
    pull, push, gives = _forces((1, 1, 2))[0]
    assert np.isclose(pull, 16) and np.isclose(push, 16) and gives == 0  # 36 / 1.5 ** 2 and 2 ** 4


def test_the_pull_grows_with_both_volumes_and_falls_with_the_distance():
    base, more_left, more_right, fatter_joint = _forces((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 0.5))
    assert np.isclose(base[0], 36 / 2 ** 2)  # the distance is a half, 1 and a half
    assert more_left[0] > base[0] and np.isclose(more_left[0], more_right[0])
    assert fatter_joint[0] < base[0]  # a fatter joint holds its volumes further apart
    assert (_forces((0, 1, 1), (1, 0, 1), (1, 1, 0))[:, [0, 2]] == 0).all()  # nothing to pull


def test_the_push_rises_steeply_and_what_is_given_is_the_unanswered_share():
    calm, twice = _forces((1, 1, 1), (1, 1, 2))
    assert calm[1] == 1 and twice[1] == 16  # twice the quanta: 16 times the push
    assert np.isclose(calm[2], GIVE * 0.01 * (9 - 1) / 9)
    assert _forces((1, 1, 3))[0, 2] == 0  # the push is the bigger: nothing gives
