"""Where a volume tears: across the direction it is pulled from both sides, else where its fluid lies widest."""
import numpy as np

from physics.engine.axiom_6_limits.definitions.overstretched import (
    half_round, least_whirl, spread, torn_across)


def test_a_pull_from_both_sides_decides_and_a_pull_from_one_side_stretches_nothing():
    cube = np.ones((6, 6, 6))
    cube[:, :, 0] = cube[:, :, 5] = 3  # the fluid lies widest along the third direction
    assert spread(cube, 2) > spread(cube, 0) == spread(cube, 1)
    assert torn_across(cube, np.zeros((3, 2))) == 2
    assert torn_across(cube, np.array([[9.0, 0], [0, 0], [0, 0]])) == 2  # one side only
    assert torn_across(cube, np.array([[0, 0], [0.1, 0.2], [0, 9.0]])) == 1  # both sides


def test_two_half_turns_about_two_directions_are_one_about_the_third():
    assert [half_round(0, way) for way in range(3)] == [1, 2, 3]
    assert half_round(1, 0) == 0 and half_round(1, 1) == 3 and half_round(3, 0) == 2


def test_the_least_whirl_is_the_same_share_of_a_cell_at_any_boxes():
    assert abs(least_whirl(6) - 10_873) < 1 and least_whirl(12) == 8 * least_whirl(6)
