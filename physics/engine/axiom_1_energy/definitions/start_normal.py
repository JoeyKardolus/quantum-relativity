"""Start normal: one standard normal number for one thing at the start, from the seed alone.

Contract: `start_normal(seed, first_pass, second_pass, index)` (Warp func; uint32, uint32,
uint32, uint64) returns a float64 standard normal by Box and Muller from the two
`draw_uniform` draws of step 0 with those passes and that index. The same seed, passes
and index give the same number on any device. Used only where the start is laid.
"""
from __future__ import annotations

import math

import warp as wp

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform

wp.set_module_options({"fuse_fp": False})

_TURN = wp.constant(wp.float64(2.0 * math.pi))
_START_STEP = wp.constant(wp.uint32(0))


@wp.func
def start_normal(seed: wp.uint32, first_pass: wp.uint32, second_pass: wp.uint32, index: wp.uint64) -> wp.float64:
    """A standard normal for one thing, by Box-Muller from its two start draws."""
    first = draw_uniform(seed, _START_STEP, first_pass, index)
    second = draw_uniform(seed, _START_STEP, second_pass, index)
    return wp.sqrt(wp.float64(-2.0) * wp.log(wp.float64(1.0) - first)) * wp.cos(_TURN * second)
