"""A joint is past its limit where twice the pull is at least the top speed squared, and not before."""
import numpy as np

from physics.engine.axiom_6_limits.definitions.collapse_limit import past_limit, top_speed

A_TURN = 1 / 25  # a step's share of a calm swirl's turn at 6 boxes across


def test_the_top_speed_is_the_joint_s_distance_in_two_steps():
    assert top_speed(2.0, A_TURN) == 25.0 and top_speed(0.5, A_TURN) == 6.25  # calm lengths a turn


def test_a_joint_is_past_its_limit_only_where_the_pull_is_deep_enough_to_hold_the_top_speed():
    edge = top_speed(1.0, A_TURN) ** 2 / 2  # the pull that just holds it, volumes 1 apart: 78.125
    assert past_limit(edge, 1.0, A_TURN) and not past_limit(0.999 * edge, 1.0, A_TURN)
    assert not past_limit(edge, 1.01, A_TURN) and past_limit(edge / 4, 0.5, A_TURN)  # closer: sooner
    pulls, aparts = np.array([1.2, 150.0, 150.0]), np.array([2.0, 13.0, 0.7])  # calm, rim, squeezed centre
    assert past_limit(pulls, aparts, A_TURN).tolist() == [False, False, True]
    assert not past_limit(1e9, 0.0, A_TURN)  # a joint with no distance read yet does not collapse
