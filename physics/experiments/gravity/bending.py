"""Light-bending probe: how close a shortest path passes a given joint.

Contract: `path_closest_approach(network, start, end, target, ruler)` finds the
shortest (length-weighted) path from `start` to `end` on `network` and returns
the smallest value of `ruler` (a fixed per-joint distance from the target,
measured once on an undisturbed network) over the joints on that path. Using a
fixed ruler means a smaller value is a path that really passes nearer the
target, not one whose lengths merely shrank.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import dijkstra

from physics.network import Network


def path_closest_approach(network: Network, start: int, end: int,
                          ruler: np.ndarray) -> float:
    """Smallest ruler value along the shortest start-to-end path."""
    _, predecessors = dijkstra(network.adjacency(), directed=False, indices=start,
                               return_predecessors=True)
    joint, closest = end, float(ruler[end])
    while joint != start and joint >= 0:
        joint = int(predecessors[joint])
        closest = min(closest, float(ruler[joint]))
    return closest
