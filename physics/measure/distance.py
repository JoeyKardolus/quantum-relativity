"""Distance read from the network: shortest paths weighted by connection length.

Contract: `distances(network, sources)` runs Dijkstra (scipy csgraph) from each
source joint over the connections, each weighted by its length. Returns an
array (len(sources), joint_count); unreachable joints are inf. Reads only the
network.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from physics.network import Network


def distances(network: Network, sources: np.ndarray | list[int]) -> np.ndarray:
    """Length-weighted shortest-path distance from each source to every joint."""
    return dijkstra(network.adjacency(), directed=False, indices=np.asarray(sources))
