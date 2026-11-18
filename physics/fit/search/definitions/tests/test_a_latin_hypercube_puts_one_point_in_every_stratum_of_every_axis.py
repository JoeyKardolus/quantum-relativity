"""A Latin hypercube of n points has exactly one point in each of the n strata of every axis, and the same seed gives the same points."""
import numpy as np

from physics.fit.search.definitions.latin_hypercube import latin_hypercube


def test_every_axis_has_one_point_a_stratum():
    points = latin_hypercube(50, 12, seed=7)
    assert points.shape == (50, 12) and points.min() >= 0.0 and points.max() < 1.0
    for axis in range(12):
        assert sorted(np.floor(points[:, axis] * 50).astype(int).tolist()) == list(range(50))


def test_the_same_seed_gives_the_same_points_and_another_seed_others():
    assert np.array_equal(latin_hypercube(20, 3, seed=1), latin_hypercube(20, 3, seed=1))
    assert not np.array_equal(latin_hypercube(20, 3, seed=1), latin_hypercube(20, 3, seed=2))


def test_no_points_is_an_empty_cube():
    assert latin_hypercube(0, 12, seed=1).shape == (0, 12)
