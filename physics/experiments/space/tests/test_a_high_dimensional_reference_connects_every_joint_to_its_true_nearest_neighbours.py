"""A high dimensional reference connects every joint to its true nearest neighbours.

WHAT IT PINS. physics/experiments/space through `high_dimensional_reference`:
each joint's shortest connection has the length of its true nearest neighbour
in the scatter, every joint has at least `neighbours` connections, and no
connection is doubled or joins a joint to itself.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. The same seeded
scatter (numpy's default_rng(seed).random((N, d)), the documented recipe) is
redrawn here and its periodic nearest-neighbour distances are worked out by
brute force over all pairs, with no k-d tree.

ONE FUNCTION. This module's one test.

WHAT IT WAS PROVED AGAINST. Lengths taken from the wrong column of the
neighbour query, so connections no longer carry the scattered distance.

GUARDS. First version of the physics package, 2026-11-14.
"""
from __future__ import annotations

import numpy as np

from physics.experiments.space.high_dimensional import high_dimensional_reference


def test_a_high_dimensional_reference_connects_every_joint_to_its_true_nearest_neighbours() -> None:
    """Shortest connection per joint equals the brute-force nearest distance."""
    joints, dimension, neighbours, seed = 400, 6, 8, 3
    network = high_dimensional_reference(joints, dimension, neighbours, seed)
    points = np.random.default_rng(seed).random((joints, dimension))
    gap = np.abs(points[:, None, :] - points[None, :, :])
    distance = np.sqrt((np.minimum(gap, 1.0 - gap) ** 2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    shortest = np.full(joints, np.inf)
    np.minimum.at(shortest, network.pairs[:, 0], network.length)
    np.minimum.at(shortest, network.pairs[:, 1], network.length)
    np.testing.assert_allclose(shortest, distance.min(axis=1), rtol=1e-9)
    assert (network.degree() >= neighbours).all()
    unordered = np.sort(network.pairs, axis=1)
    assert len(np.unique(unordered, axis=0)) == network.connection_count
