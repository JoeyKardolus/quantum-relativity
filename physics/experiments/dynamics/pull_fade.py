"""Pull around a packed lump, by how many connections lie in between.

Contract: `pull_by_hops(seed)` builds a 3D spatial reference network
(JOINTS joints), packs one lump (connections within LUMP_HOPS connections of
joint 0 get (1 + AMPLITUDE) times the energy and the matching shorter length),
computes the v2 pull on the packed and on the untouched network, and returns
{"hops": 0..MAX_HOPS, "excess": mean pull added by the lump at each hop count
(hop = fewest connections from joint 0), "background": median pull without the
lump}. Hop counts read the network's connections only.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.sparse.csgraph import shortest_path

from physics.network import spatial_reference
from physics.rules import pull

JOINTS = 2000
LUMP_HOPS = 1
AMPLITUDE = 4.0
MAX_HOPS = 8


def pull_by_hops(seed: int) -> dict:
    """Mean extra pull per hop shell around a packed lump, and the background pull."""
    network = spatial_reference(JOINTS, 3, seed=seed)[0]
    hops = shortest_path(network.adjacency(np.ones(network.connection_count)), unweighted=True,
                         indices=0)
    near = hops[network.pairs].min(axis=1) < LUMP_HOPS
    factor = np.where(near, 1.0 + AMPLITUDE, 1.0)
    packed = replace(network, energy=network.energy * factor, length=network.length / factor)
    excess = pull(packed, packed.energy, packed.length) - pull(network, network.energy, network.length)
    shells = np.arange(MAX_HOPS + 1)
    return {"hops": shells.tolist(),
            "excess": [float(excess[hops == shell].mean()) for shell in shells],
            "background": float(np.median(pull(network, network.energy, network.length)))}
