"""A clustered reference labels each cluster's joints, which mostly connect among themselves.

WHAT IT PINS. physics/experiments/expansion through `clustered_reference`: one
label per joint (-1 for sparse space, 0..k-1 per cluster, cluster_size each),
and a cluster joint's connections mostly stay inside its own cluster.

HOW THE EXPECTED ANSWER IS DERIVED WITHOUT THE CODE UNDER TEST. A cluster of
300 points of width 0.02 in a unit cube holding 2000 uniform points is about
10,000 times denser than the space around it, so its joints' nearest
neighbours are almost all in the same cluster; with labels shuffled between
clusters the share would be about 1 / clusters = 0.25.

ONE FUNCTION. This module's one test.

WHAT IT WAS PROVED AGAINST. Labels tiled across clusters instead of repeated
per cluster, so they no longer name the scatter a joint came from.

GUARDS. First version of the physics package, 2026-11-14.
"""
from __future__ import annotations

import numpy as np

from physics.experiments.expansion.clustered_reference import clustered_reference


def test_a_clustered_reference_labels_each_clusters_joints_which_mostly_connect_among_themselves() -> None:
    """Label counts per scatter, and same-cluster share of cluster connections above 0.9."""
    network, labels = clustered_reference(2000, 4, 300, 0.02, 8, seed=5)
    assert network.joint_count == len(labels) == 3200
    assert (labels == -1).sum() == 2000
    assert all((labels == cluster).sum() == 300 for cluster in range(4))
    heads, tails = labels[network.pairs[:, 0]], labels[network.pairs[:, 1]]
    in_cluster = (heads >= 0) | (tails >= 0)
    assert np.mean(heads[in_cluster] == tails[in_cluster]) > 0.9
