"""Scene "collapse": one very heavy lump whose pull should pass the push limit.

Contract: `collapse(seed, joint_count)` builds an even reference ball, puts one
lump (radius LUMP_RADIUS, amplitude AMPLITUDE) on the centre joint and runs
every rule with the balancing drift of the ball before the lump
(`scene_rules`), so the squeeze inside the lump is meant to pass the push limit
and joints collapse. Whether they do is read off the recording, never assumed.
"""
from __future__ import annotations

from physics.experiments.gravity.lumps import add_lumps
from physics.measure import mean_length
from physics.viewer.reference_ball import reference_ball
from physics.viewer.scene import Scene
from physics.viewer.scene_rules import scene_rules

LUMP_RADIUS = 2.0       # in mean lengths
AMPLITUDE = 20.0


def collapse(seed: int = 1, joint_count: int = 3000) -> Scene:
    """One heavy lump at the centre, every rule on, drift balancing the even ball."""
    ball = reference_ball(joint_count, seed)
    network = add_lumps(ball, [0], LUMP_RADIUS * mean_length(ball), AMPLITUDE)
    return Scene("collapse", "Joint collapse", network, scene_rules(ball, seed), (0,),
                 "Watch the heavy lump in the middle: if its pull squeezes joints past "
                 "their push limit they collapse, and the lump should shrink and run away.", seed)
