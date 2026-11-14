"""Display-only 2D layout from network distances.

Contract: `display_layout(network, joints)` returns an array (len(joints), 2)
of drawing positions for the given joints: classical multidimensional scaling
of their length-weighted shortest-path distances (Dijkstra on the whole
network), so drawn distances follow network distances in the units of the
network's lengths. Unreachable pairs count as the largest finite distance. The
positions are for one picture only and are never stored on the Network.
"""
from __future__ import annotations

import numpy as np

from physics.measure import distances
from physics.network import Network


def display_layout(network: Network, joints: np.ndarray) -> np.ndarray:
    """Classical MDS of the joints' network distances, two dimensions."""
    table = distances(network, joints)[:, joints]
    table = np.where(np.isfinite(table), table, table[np.isfinite(table)].max())
    squared = ((table + table.T) / 2) ** 2
    centred = squared - squared.mean(axis=0) - squared.mean(axis=1)[:, None] + squared.mean()
    values, vectors = np.linalg.eigh(-centred / 2)
    top = np.argsort(values)[::-1][:2]
    positions = np.zeros((len(joints), 2))
    positions[:, :len(top)] = vectors[:, top] * np.sqrt(np.clip(values[top], 0.0, None))
    return positions
