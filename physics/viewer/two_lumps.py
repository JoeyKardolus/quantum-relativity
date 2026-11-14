"""Scene "two-lumps": two lumps of extra energy in a ball of 3D space.

Contract: `two_lumps(seed, joint_count)` builds a reference ball, puts two
lumps (`add_lumps`, radius LUMP_RADIUS, amplitude AMPLITUDE) on joints about
OFFSET mean connection lengths from the centre and as far from each other as
the ball allows, and runs energy flow, energy-length, attraction (scaled by
the median pull at the start, so the background squeeze per step is ATTRACTION) and joint push with a
finite push limit set well above the push this scene should need.
"""
from __future__ import annotations

import numpy as np

from physics.experiments.gravity.lumps import add_lumps
from physics.measure import distances, mean_length
from physics.network import Network
from physics.rules.pull import pull
from physics.viewer.reference_ball import reference_ball
from physics.viewer.rule_settings import rule_settings
from physics.viewer.scene import Scene

OFFSET = 4.0            # lump centre distance from the ball centre, in mean lengths
LUMP_RADIUS = 2.0       # in mean lengths
AMPLITUDE = 4.0         # lump connections hold 1 + AMPLITUDE times their energy
ENERGY_FLOW = 0.05
ATTRACTION = 0.01       # shortening per step at the median pull (the background squeeze)
JOINT_PUSH = 0.5
PUSH_LIMIT = 1.0       # 200 times the background push: not meant to be reached


def two_lumps(seed: int = 1, joint_count: int = 3000) -> Scene:
    """Two lumps some distance apart, attraction and joint push on, no collapse expected."""
    ball = reference_ball(joint_count, seed)
    unit = mean_length(ball)
    from_centre = distances(ball, [0])[0] / unit
    first = int(np.argmin(np.abs(from_centre - OFFSET)))
    from_first = distances(ball, [first])[0]
    ring = np.flatnonzero(np.abs(from_centre - OFFSET) < 0.5)
    second = int(ring[np.argmax(from_first[ring])])
    network = add_lumps(ball, [first, second], LUMP_RADIUS * unit, AMPLITUDE)
    settings, missing = rule_settings({
        "energy_length": 1.0, "energy_flow": ENERGY_FLOW,
        "attraction": ATTRACTION / _background_pull(ball),
        "joint_push": JOINT_PUSH, "push_limit": PUSH_LIMIT})
    return Scene("two-lumps", "Two lumps", network, settings, missing, (first, second),
                 "Watch whether the two warm lumps drift toward each other or only shrink "
                 "where they sit, while the joints push back on the squeeze.", seed)


def _background_pull(network: Network) -> float:
    return float(np.median(pull(network, network.energy, network.length)))
