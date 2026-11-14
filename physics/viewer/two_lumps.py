"""Scene "two-lumps": two lumps of extra energy in an even ball of 3D space.

Contract: `two_lumps(seed, joint_count)` builds an even reference ball, puts
two lumps (`add_lumps`, radius LUMP_RADIUS, amplitude AMPLITUDE) on joints
about OFFSET mean connection lengths from the centre and as far from each other
as the ball allows, and runs every rule with the balancing drift of the ball
before the lumps (`scene_rules`).
"""
from __future__ import annotations

import numpy as np

from physics.experiments.gravity.lumps import add_lumps
from physics.measure import distances, mean_length
from physics.viewer.reference_ball import reference_ball
from physics.viewer.scene import Scene
from physics.viewer.scene_rules import scene_rules

OFFSET = 4.0            # lump centre distance from the ball centre, in mean lengths
LUMP_RADIUS = 2.0       # in mean lengths
AMPLITUDE = 4.0         # lump connections hold 1 + AMPLITUDE times their energy


def two_lumps(seed: int = 1, joint_count: int = 3000) -> Scene:
    """Two lumps some distance apart, every rule on, drift balancing the even ball."""
    ball = reference_ball(joint_count, seed)
    unit = mean_length(ball)
    from_centre = distances(ball, [0])[0] / unit
    first = int(np.argmin(np.abs(from_centre - OFFSET)))
    from_first = distances(ball, [first])[0]
    ring = np.flatnonzero(np.abs(from_centre - OFFSET) < 0.5)
    second = int(ring[np.argmax(from_first[ring])])
    network = add_lumps(ball, [first, second], LUMP_RADIUS * unit, AMPLITUDE)
    return Scene("two-lumps", "Two lumps", network, scene_rules(ball, seed), (first, second),
                 "Watch whether the two warm lumps close in on each other faster than the space "
                 "around them shrinks; the camera stays put, so a shrink of all of space shows as "
                 "the whole drawing getting smaller.", seed)
