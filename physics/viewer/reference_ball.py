"""The starting space for every scene: a ball of a 3D reference network.

Contract: `reference_ball(joint_count, seed)` scatters TORUS_FACTOR times
joint_count points on the flat unit 3-torus (`spatial_reference`, points then
dropped), and carves the joint_count joints nearest joint 0 by network
distance (`carve_ball`), so the scene has no wrap-around, then makes its
energy even: every connection at the ball's mean length, energy by the
energy-length rule. Even is the background the balancing drift is calibrated
on. Joint 0 of the result is the ball's centre.
"""
from __future__ import annotations

import numpy as np

from physics.network import Network, spatial_reference
from physics.viewer.region import carve_ball

TORUS_FACTOR = 5


def reference_ball(joint_count: int, seed: int) -> Network:
    """An even ball of joint_count joints from a 3D reference network, centre joint 0."""
    torus, known = spatial_reference(TORUS_FACTOR * joint_count, 3, seed=seed)
    ball = carve_ball(torus, 0, joint_count, known["side"] / 2)
    return Network.from_lengths(ball.joint_count, ball.pairs,
                                np.full(ball.connection_count, ball.length.mean()))
