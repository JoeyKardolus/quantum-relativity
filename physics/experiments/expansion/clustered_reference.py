"""Clustered spatial reference: sparse 3D space holding a few dense clusters.

Contract: `clustered_reference(sparse_count, cluster_count, cluster_size, spread,
neighbours, seed)` scatters sparse_count points uniformly in the unit 3-torus plus
cluster_count Gaussian blobs (cluster_size points each, width `spread`), connects
every point to its `neighbours` nearest (periodic), sets length to the scattered
distance (energy by the energy-length rule) and DISCARDS the points. Returns the
Network and a label per joint (-1 sparse, 0.. cluster index): which scatter a
joint came from, a membership tag used only to choose whom to measure, never a
coordinate.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from physics.network import Network


def clustered_reference(sparse_count: int, cluster_count: int, cluster_size: int,
                        spread: float, neighbours: int, seed: int) -> tuple[Network, np.ndarray]:
    """k-nearest network over uniform space plus Gaussian clusters; points are dropped."""
    rng = np.random.default_rng(seed)
    centres = rng.random((cluster_count, 3))
    blobs = [(centre + spread * rng.standard_normal((cluster_size, 3))) % 1.0 for centre in centres]
    points = np.vstack([rng.random((sparse_count, 3)), *blobs])
    labels = np.concatenate([np.full(sparse_count, -1),
                             np.repeat(np.arange(cluster_count), cluster_size)])
    joint_count = len(points)
    tree = cKDTree(points, boxsize=1.0)
    distance, index = tree.query(points, k=neighbours + 1)
    heads = np.repeat(np.arange(joint_count), neighbours)
    tails = index[:, 1:].ravel()
    low, high = np.minimum(heads, tails), np.maximum(heads, tails)
    keys, first = np.unique(low * joint_count + high, return_index=True)
    pairs = np.stack([keys // joint_count, keys % joint_count], axis=1)
    network = Network.from_lengths(joint_count, pairs, distance[:, 1:].ravel()[first])
    del points, tree, centres, blobs
    return network, labels
