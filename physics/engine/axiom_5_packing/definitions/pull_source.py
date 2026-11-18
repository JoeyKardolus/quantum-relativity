"""Pull source: more packed pulls harder, and only plus pulls.

Contract: `pull_source(kind, packing, pull_strength)` (Warp func; int32, float64,
float64) returns `4 pi G * packing` as float64 where the sign law says the kind makes a
pull (an energy volume), else 0: a joint carries the pull but adds none.
"""
from __future__ import annotations

import math

import warp as wp

from physics.engine.axiom_2_swirls.definitions.sign_law import makes_pull

wp.set_module_options({"fuse_fp": False})

_FOUR_PI = wp.constant(wp.float64(4.0 * math.pi))


@wp.func
def pull_source(kind: wp.int32, packing: wp.float64, pull_strength: wp.float64) -> wp.float64:
    """What a cell adds to the pull: 4 pi G times its packing, for plus energy only."""
    if makes_pull(kind):
        return _FOUR_PI * pull_strength * packing
    return wp.float64(0.0)
