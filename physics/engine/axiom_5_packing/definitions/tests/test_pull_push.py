"""Two volumes stand their half widths and their joint's thickness apart, never closer; the pull
across falls with it and grows with the pull felt at the two volumes; the push rises steeply."""
import numpy as np
import warp as wp

from physics.engine.axiom_5_packing.definitions.pull_push import (
    GIVE, PULL, apart_of, given, pull_across, push_back)


@wp.kernel
def _read(held: wp.array(dtype=wp.vec3d), felt: wp.float64, found: wp.array(dtype=wp.vec4d)):
    left, right, joint = held[wp.tid()][0], held[wp.tid()][1], held[wp.tid()][2]
    apart = apart_of(left, right, joint)
    pull, push = pull_across(felt * left, felt * right, apart), push_back(joint)
    gives = given(wp.vec4d(pull, push, pull, pull), wp.float64(0.01))
    found[wp.tid()] = wp.vec4d(pull, push, gives, apart)


def _forces(*held: tuple, felt: float = 1.0) -> np.ndarray:
    """Pull, push, what is given in a hundredth of a turn and the distance, for each (left, right,
    joint) quanta; each volume feels a pull of `felt` times its own quanta."""
    found = wp.zeros(len(held), dtype=wp.vec4d, device="cpu")
    wp.launch(_read, dim=len(held), device="cpu",
              inputs=[wp.array(held, dtype=wp.vec3d, device="cpu"), wp.float64(felt), found])
    return found.numpy()


def test_the_distance_is_half_a_width_the_joint_s_thickness_and_half_the_other_width():
    fresh, squeezed, uneven = _forces((1, 1, 1), (1, 1, 2), (2, 0.5, 1))[:, 3]
    assert np.isclose(fresh, 0.5 + 1 + 0.5) and np.isclose(squeezed, 0.5 + 0.5 + 0.5)
    assert np.isclose(uneven, 0.25 + 1 + 1)  # sizes 0.5 and 2, a joint 1 thick


def test_two_joined_volumes_never_stand_closer_than_their_widths_however_squeezed():
    held = [(left, right, joint) for left in (0.02, 1, 50) for right in (0.02, 1, 50)
            for joint in (0.01, 1, 1e6)]  # joints from fresh and fat to squeezed a million times
    apart = _forces(*held)[:, 3]
    widths = np.array([0.5 / left + 0.5 / right for left, right, _joint in held])
    assert (apart > widths).all() and np.allclose(apart - widths, [1 / one[2] for one in held])


def test_two_calm_volumes_hold_their_joint_as_thick_as_they_are_wide():
    pull, push, gives, apart = _forces((1, 1, 1.017), felt=1.2)[0]  # alone, each feels 1.2
    assert abs(apart - 1.983) < 0.001 and abs(pull - push) < 0.01 * push


def test_the_pull_grows_with_both_volumes_and_falls_with_the_distance():
    base, more_left, more_right, fatter_joint = _forces((1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 0.5))
    assert np.isclose(base[0], PULL / base[3] ** 2)
    assert more_left[0] > base[0] and np.isclose(more_left[0], more_right[0])
    assert fatter_joint[0] < base[0]  # a fatter joint holds its volumes further apart
    assert (_forces((1, 1, 1), felt=0.0)[:, [0, 2]] == 0).all()  # no pull felt: no squeeze


def test_the_push_rises_steeply_and_what_is_given_is_the_unanswered_share():
    calm, twice = _forces((1, 1, 1), (1, 1, 2), felt=2.0)
    assert calm[1] == 1 and twice[1] == 16  # twice the quanta: 16 times the push
    assert calm[0] > 1 and np.isclose(calm[2], GIVE * 0.01 * (calm[0] - 1) / calm[0])
    assert twice[2] == 0  # the push is the bigger: nothing gives
