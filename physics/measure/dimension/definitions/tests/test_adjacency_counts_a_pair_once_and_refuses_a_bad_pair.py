"""A graph is which nodes touch: a pair listed twice counts once, both ways; a bad pair is refused."""
import numpy as np
import pytest

from physics.measure.dimension.definitions.adjacency import adjacency


def test_a_pair_listed_twice_counts_once_both_ways():
    matrix = adjacency(np.array([0, 1, 0]), np.array([1, 0, 2]), 3)
    assert matrix.toarray().tolist() == [[0, 1, 1], [1, 0, 0], [1, 0, 0]]


@pytest.mark.parametrize("low, high", [([1], [1]), ([0], [3]), ([-1], [0]), ([0, 1], [1])])
def test_a_self_pair_an_id_out_of_range_and_unequal_lists_are_refused(low, high):
    with pytest.raises(ValueError):
        adjacency(np.array(low), np.array(high), 3)
