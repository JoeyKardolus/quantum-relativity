"""A random network with no geometry: the space test's start state.

Contract: `random_network(joint_count, mean_degree, energy, seed)` joins random
pairs of joints (Erdos-Renyi style, no self-connections, no duplicates) until
the mean degree is about `mean_degree`; every connection gets the same energy
(length by the energy-length rule). Nothing about it is spatial.
"""
from __future__ import annotations

import numpy as np

from physics.network.network import Network
from physics.network.energy_length import energy_length


def random_network(joint_count: int, mean_degree: float = 6.0, energy: float = 1.0,
                   seed: int = 0) -> Network:
    """Uniformly random connections between joint_count joints, all of equal energy."""
    rng = np.random.default_rng(seed)
    wanted = int(round(joint_count * mean_degree / 2))
    draws = rng.integers(0, joint_count, size=(int(wanted * 1.2) + 16, 2))
    draws = draws[draws[:, 0] != draws[:, 1]]
    low, high = draws.min(axis=1), draws.max(axis=1)
    _, first = np.unique(low * joint_count + high, return_index=True)
    first = np.sort(first)[:wanted]
    pairs = np.stack([low[first], high[first]], axis=1).astype(np.int64)
    energies = np.full(len(pairs), float(energy))
    return Network(joint_count, pairs, energies, np.full(len(pairs), energy_length(energy)))
