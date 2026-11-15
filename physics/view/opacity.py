"""Opacity: how much light a box adds, on a fixed log scale of its density contrast.

Contract: `opacity(contrast, boxes_deep)` is a Warp function (callable inside a Warp
kernel, on any device) returning a float32 in [0, 1], the light one box adds to
the pixels its dot covers (drawn additively, so light from boxes behind each other
sums). contrast is a box's energy density over the start's mean density of its own kind
(`start_reference.py`; Planck boxes are all one size, so this is a ratio of
energies); boxes_deep is the expected number of boxes along one line of sight
(the cube root of the start's occupied box count). The scale is fixed and log-linear: the log of the opacity is linear in
log contrast, reaching 1 at contrast PACKED, and at contrast 1 (the start's mean)
it is AVERAGE_GLOW / boxes_deep, so a whole line of sight of average boxes sums
to only AVERAGE_GLOW: an even universe is nearly black, and only what is denser
than average glows. Thinner than average falls further on the same line (half
the mean at side 256 is about 1e-5, nothing to see): empty and thin space is
drawn as nothing, not as a haze. Above PACKED it stays 1. A non-finite or
non-positive contrast gives 0, except +inf, which gives 1. The scale never moves
with the data: not rank-based, never re-normalised per frame, so the same
contrast always gives the same light and a denser box is never fainter than a
thinner one. This is the only copy of the rule; the GPU gather (`box_points.py`)
calls it.
"""
from __future__ import annotations

import math

import warp as wp

AVERAGE_GLOW = 0.02
PACKED = 16.0

_LOG_PACKED = math.log(PACKED)
_LOG_AVERAGE_GLOW = math.log(AVERAGE_GLOW)


@wp.func
def opacity(contrast: wp.float32, boxes_deep: wp.float32) -> wp.float32:
    """The light one box adds, from its density over the start's mean density."""
    light = wp.float32(0.0)
    if wp.isinf(contrast) and contrast > wp.float32(0.0):
        light = wp.float32(1.0)
    elif wp.isfinite(contrast) and contrast > wp.float32(0.0):
        log_floor = wp.float32(_LOG_AVERAGE_GLOW) - wp.log(wp.max(boxes_deep, wp.float32(1.0)))
        depth = wp.min(wp.log(contrast) / wp.float32(_LOG_PACKED), wp.float32(1.0))
        light = wp.exp(log_floor * (wp.float32(1.0) - depth))
    return light
