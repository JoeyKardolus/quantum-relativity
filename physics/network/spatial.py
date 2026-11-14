"""Spatial reference network: points scattered in a flat torus, then forgotten.

Contract: `spatial_reference(joint_count, dimension, neighbours, seed)` scatters
joint_count points uniformly in the unit d-torus (d = 1..4), connects each to
its `neighbours` nearest (periodic distance; the union of both directions, so
a joint can have more), sets each connection's length to the scattered distance
(energy by the energy-length rule) and DISCARDS the points. Returns the Network
and the known answers {"dimension", "density", "side"} for comparison only.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from physics.network.network import Network


def spatial_reference(joint_count: int, dimension: int, neighbours: int | None = None,
                      seed: int = 0) -> tuple[Network, dict]:
    """A k-nearest-neighbour network on a flat d-torus; coordinates are not kept."""
    if not 1 <= dimension <= 4:
        raise ValueError("dimension must be 1..4")
    neighbours = neighbours or 2 * dimension + 4
    rng = np.random.default_rng(seed)
    points = rng.random((joint_count, dimension))
    tree = cKDTree(points, boxsize=1.0)
    distance, index = tree.query(points, k=neighbours + 1)
    heads = np.repeat(np.arange(joint_count), neighbours)
    tails = index[:, 1:].ravel()
    lengths = distance[:, 1:].ravel()
    low, high = np.minimum(heads, tails), np.maximum(heads, tails)
    keys, first = np.unique(low * joint_count + high, return_index=True)
    pairs = np.stack([keys // joint_count, keys % joint_count], axis=1)
    network = Network.from_lengths(joint_count, pairs, lengths[first])
    del points, tree
    return network, {"dimension": dimension, "density": float(joint_count), "side": 1.0}
