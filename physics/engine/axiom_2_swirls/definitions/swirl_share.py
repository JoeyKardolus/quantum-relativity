"""Swirl share: how much of a swirl's energy each box holds; a whirl holds its own core open.

Contract: `swirl_share(axis, across, along, up)` (Warp func; int32s) returns as float64
`m(first) ** 2 + m(second) ** 2`, the size squared of the standing swirl's wave
(swirl_phase.py; `first`, `second` and `m` as there) in box (across, along, up): 0.13
beside the axis, 1.87 in the four corners across it, and exactly 1 on average over the
block. The same for both senses and in every layer along the axis. `swirl_share_host`
is the host twin.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_2_swirls.definitions.swirl_phase import swirl_mode, swirl_mode_host

wp.set_module_options({"fuse_fp": False})


@wp.func
def swirl_share(axis: wp.int32, across: wp.int32, along: wp.int32, up: wp.int32) -> wp.float64:
    """The share of its cell's mean a box of a standing swirl holds."""
    first = swirl_mode(across)
    second = swirl_mode(along)
    if axis == 0:
        first = swirl_mode(along)
        second = swirl_mode(up)
    elif axis == 1:
        second = swirl_mode(up)
    return first * first + second * second


def swirl_share_host(axis: int, across: int, along: int, up: int) -> float:
    """The share of its cell's mean a box of a standing swirl holds (host)."""
    first, second = (swirl_mode_host(value) for value in ((along, up), (across, up), (across, along))[int(axis)])
    return first * first + second * second
