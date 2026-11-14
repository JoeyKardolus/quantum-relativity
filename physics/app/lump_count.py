"""How many lumps the network holds right now, read from the network only.

Contract: `lump_count(network)` gives each joint its local energy: the mean
energy of its connections, or for a joint with no connection the energy it
holds (`held`, left by a whole cluster collapsing). A joint is hot when its
local energy is at least LUMP_FACTOR (1.5) times the mean connection energy.
A lump is a group of hot joints connected through connections whose both
joints are hot; a single hot joint is a lump of one. Returns the number of
lumps.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from physics.network import Network

LUMP_FACTOR = 1.5


def lump_count(network: Network) -> int:
    """Connected groups of joints whose local energy is well above the mean."""
    if not network.connection_count:
        return 0
    degree = network.degree()
    summed = np.bincount(network.pairs.ravel(), np.repeat(network.energy, 2), network.joint_count)
    local = summed / np.maximum(degree, 1)
    if network.held is not None:
        local = np.where(degree == 0, network.held, local)
    hot = local >= LUMP_FACTOR * float(network.energy.mean())
    if not hot.any():
        return 0
    inside = network.pairs[hot[network.pairs[:, 0]] & hot[network.pairs[:, 1]]]
    joined = sp.coo_matrix((np.ones(len(inside)), (inside[:, 0], inside[:, 1])),
                           shape=(network.joint_count, network.joint_count))
    _, labels = connected_components(joined, directed=False)
    return int(len(np.unique(labels[hot])))
