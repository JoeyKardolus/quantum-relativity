"""Spectral dimension where the walk is known: on a line the return chance falls as t to the minus a half."""
import numpy as np
import pytest

from physics.measure.dimension.definitions.adjacency import adjacency
from physics.measure.dimension.logic.spectral_dimension import spectral_dimension

RING = 2001


def _ring():
    return adjacency(np.arange(RING), (np.arange(RING) + 1) % RING, RING)


def test_a_walk_on_a_long_ring_gives_dimension_one_within_two_percent():
    measured = spectral_dimension(_ring(), [0], np.arange(10, 101))
    assert abs(measured["dimension"] - 1.0) <= 0.02


def test_after_one_lazy_step_the_walker_is_home_with_chance_one_half_and_after_two_three_eighths():
    measured = spectral_dimension(_ring(), [0, 7], [1, 2])
    assert measured["return_probability"] == pytest.approx([0.5, 0.375])


def test_a_source_with_no_neighbour_is_refused():
    with pytest.raises(ValueError):
        spectral_dimension(adjacency(np.array([0]), np.array([1]), 3), [2], [1, 2])
