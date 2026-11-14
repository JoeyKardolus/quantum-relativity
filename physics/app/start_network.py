"""The live app's starting space: an even 3D ball, and the same ball made slightly uneven.

Contract: `start_network(joints, seed, unevenness)` scatters TORUS_FACTOR times
`joints` points on the flat unit 3-torus with NEIGHBOURS nearest connections
(`spatial_reference`, points then dropped), carves the `joints` joints nearest
joint 0 by network distance (`carve_ball`, so nothing wraps around), and makes
the energy even: every connection at the ball's mean length. Returns
(even, uneven): `even` is the background the balancing drift is calibrated
on; `uneven` holds energy background * (1 + unevenness * NOISE * z) per
connection, z standard normal from the seed, floored at FLOOR times the
background, length by the energy-length rule. No lump is placed.
"""
from __future__ import annotations

import numpy as np

from physics.network import Network, energy_length, spatial_reference
from physics.viewer.region import carve_ball

TORUS_FACTOR = 5
NEIGHBOURS = 8
NOISE = 0.1     # unevenness 1 gives a 10% standard deviation in connection energy
FLOOR = 0.1


def start_network(joints: int, seed: int, unevenness: float) -> tuple[Network, Network]:
    """Even ball and its slightly uneven copy, both without coordinates."""
    torus, known = spatial_reference(TORUS_FACTOR * joints, 3, NEIGHBOURS, seed=seed)
    ball = carve_ball(torus, 0, joints, known["side"] / 2)
    even = Network.from_lengths(ball.joint_count, ball.pairs,
                                np.full(ball.connection_count, ball.length.mean()))
    noise = np.random.default_rng(seed).standard_normal(even.connection_count)
    energy = even.energy * np.maximum(1 + unevenness * NOISE * noise, FLOOR)
    return even, Network(even.joint_count, even.pairs, energy, energy_length(energy))
