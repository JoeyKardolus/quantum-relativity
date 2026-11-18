"""The range rule by hand: a quarter of the farthest distance, never under 3."""
import pytest

from physics.measure.dimension.definitions.range_rule import range_rule


def test_a_graph_100_steps_deep_gives_radii_2_to_25_and_times_62_to_625():
    radii, times = range_rule(100)
    assert radii.tolist() == list(range(2, 26))
    assert (times[0], times[-1], len(times)) == (62, 625, 564)


def test_a_small_graph_still_gets_a_reach_of_three():
    radii, times = range_rule(4)
    assert radii.tolist() == [2, 3] and times.tolist() == [2, 3, 4, 5, 6, 7, 8, 9]


def test_a_graph_with_no_far_node_is_refused():
    with pytest.raises(ValueError):
        range_rule(0)
