"""A spatial reference network in more than four dimensions (the core builder stops at 4).

Contract: `high_dimensional_reference(joint_count, dimension, neighbours, seed)`
scatters points uniformly in the unit d-torus, connects each to its
`neighbours` nearest (periodic distance, union of both directions), sets each
connection's length to the scattered distance and DISCARDS the points: the same
recipe as `physics.network.spatial_reference`, without its d <= 4 limit.
Returns the Network only; the known dimension is the caller's `dimension`.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from physics.network import Network


def high_dimensional_reference(joint_count: int, dimension: int, neighbours: int,
                               seed: int = 0) -> Network:
    """k-nearest-neighbour network on a flat d-torus; coordinates are not kept."""
    rng = np.random.default_rng(seed)
    points = rng.random((joint_count, dimension))
    distance, index = cKDTree(points, boxsize=1.0).query(points, k=neighbours + 1)
    heads = np.repeat(np.arange(joint_count), neighbours)
    tails = index[:, 1:].ravel()
    lengths = distance[:, 1:].ravel()
    low, high = np.minimum(heads, tails), np.maximum(heads, tails)
    keys, first = np.unique(low * joint_count + high, return_index=True)
    pairs = np.stack([keys // joint_count, keys % joint_count], axis=1)
    return Network.from_lengths(joint_count, pairs, lengths[first])
