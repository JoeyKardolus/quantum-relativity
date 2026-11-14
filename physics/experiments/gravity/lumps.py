"""Lumps of extra energy on a network, placed without coordinates.

Contract: `add_lumps(network, centres, radius, amplitude)` returns a new Network
in which every connection with an end within network distance `radius` of a
centre joint has its energy multiplied by (1 + amplitude) and its length divided
by the same factor, so energy * length is unchanged (the energy-length rule
holds from the start). Distances are length-weighted shortest paths.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.measure import distances
from physics.network import Network


def add_lumps(network: Network, centres: list[int], radius: float,
              amplitude: float) -> Network:
    """Concentrate extra energy on the connections near each centre joint."""
    near = (distances(network, centres) <= radius).any(axis=0)
    touched = near[network.pairs[:, 0]] | near[network.pairs[:, 1]]
    factor = np.where(touched, 1.0 + amplitude, 1.0)
    return replace(network, energy=network.energy * factor, length=network.length / factor)
