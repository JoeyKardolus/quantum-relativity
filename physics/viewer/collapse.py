"""Scene "collapse": one very heavy lump whose pull should pass the push limit.

Contract: `collapse(seed, joint_count)` builds a reference ball, puts one lump
(radius LUMP_RADIUS, amplitude AMPLITUDE) on the centre joint and runs
energy-length, attraction (scaled like two-lumps) and joint push with a low
push limit, so the squeeze inside the lump is meant to pass the limit and
joints collapse.
Whether they do is read off the recording, never assumed.
"""
from __future__ import annotations

import numpy as np

from physics.experiments.gravity.lumps import add_lumps
from physics.measure import mean_length
from physics.network import Network
from physics.rules.pull import pull
from physics.viewer.reference_ball import reference_ball
from physics.viewer.rule_settings import rule_settings
from physics.viewer.scene import Scene

LUMP_RADIUS = 2.0       # in mean lengths
AMPLITUDE = 20.0
ATTRACTION = 0.01       # shortening per step at the median pull (the background squeeze)
JOINT_PUSH = 0.5
PUSH_LIMIT = 0.05      # 10 times the background push


def collapse(seed: int = 1, joint_count: int = 3000) -> Scene:
    """One heavy lump at the centre, attraction strong enough to break joints."""
    ball = reference_ball(joint_count, seed)
    network = add_lumps(ball, [0], LUMP_RADIUS * mean_length(ball), AMPLITUDE)
    settings, missing = rule_settings({
        "energy_length": 1.0,
        "attraction": ATTRACTION / _background_pull(ball),
        "joint_push": JOINT_PUSH, "push_limit": PUSH_LIMIT})
    return Scene("collapse", "Joint collapse", network, settings, missing, (0,),
                 "Watch the heavy lump in the middle: if its pull squeezes joints past "
                 "their push limit they collapse, and the lump should shrink and run away.", seed)


def _background_pull(network: Network) -> float:
    return float(np.median(pull(network, network.energy, network.length)))
