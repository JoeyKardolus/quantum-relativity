"""A finite region cut out of a reference network, chosen by network distance only.

Contract: `carve_ball(network, centre, joint_count, limit)` keeps the joint_count
joints nearest to `centre` by length-weighted path distance, with every
connection between two kept joints (energy and length unchanged), renumbered
0..joint_count-1 in order of distance (the centre becomes joint 0). A
reference network on a torus wraps around, which no 3D display can show; a
ball smaller than half the torus does not. Raises ValueError when the ball's
radius reaches `limit` (pass half the torus side), because then it would
touch itself across the wrap.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.measure import distances
from physics.network import Network


def carve_ball(network: Network, centre: int, joint_count: int, limit: float) -> Network:
    """The joint_count joints nearest the centre, as a network of their own."""
    reach = distances(network, [centre])[0]
    kept = np.argsort(reach, kind="stable")[:joint_count]
    if reach[kept[-1]] >= limit:
        raise ValueError(f"ball radius {reach[kept[-1]]:.3f} reaches the wrap limit {limit}")
    renumber = np.full(network.joint_count, -1)
    renumber[kept] = np.arange(joint_count)
    inside = (renumber[network.pairs[:, 0]] >= 0) & (renumber[network.pairs[:, 1]] >= 0)
    return replace(network, joint_count=joint_count, pairs=renumber[network.pairs[inside]],
                   energy=network.energy[inside], length=network.length[inside],
                   charge=network.charge[kept], causal=None)
