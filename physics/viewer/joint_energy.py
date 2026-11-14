"""Energy held at each joint: half the energy of every connection that meets there.

Contract: `joint_energy(network)` returns one value per joint, the same region
energy `rules/pull.py` uses (each connection's energy split evenly between its
two joints), so the values sum to the network's total energy.
"""
from __future__ import annotations

import numpy as np

from physics.network import Network


def joint_energy(network: Network) -> np.ndarray:
    """Half the summed energy of each joint's connections."""
    return np.bincount(network.pairs.ravel(), np.repeat(network.energy, 2),
                       network.joint_count) / 2
