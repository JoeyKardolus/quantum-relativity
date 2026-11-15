"""Opacity: how solid a box's colour is, on a fixed log scale of its energy against the start.

Contract: `opacity(ratio)` is a Warp function (callable inside a Warp kernel, on any
device) returning a float32 in [FAINTEST, 1]. ratio is a box's energy of one kind
over the start's mean of that kind (`start_reference.py`); boxes are all one size,
so this is the ratio of energy densities. It is read on a fixed log scale:
ratio <= THINNEST gives FAINTEST, ratio >= PACKED gives 1, and in between the log
of the opacity is linear in log ratio. At ratio 1 (the start's mean) a box is near
invisible (about 0.025), so thin energy is a faint haze and a mass that packs
glows as it packs. The scale never moves with the data: not rank-based, never
re-normalised per frame, so the same energy always gives the same opacity and a
denser box is never fainter than a thinner one. A non-finite or non-positive ratio
counts as thinnest. This is the only copy of the rule; the GPU sampler
(`box_points.py`) calls it.
"""
from __future__ import annotations

import math

import warp as wp

FAINTEST = 0.01
THINNEST = 0.5
PACKED = 16.0

_LOG_FAINTEST = math.log(FAINTEST)
_LOG_THINNEST = math.log(THINNEST)
_LOG_SPAN = math.log(PACKED) - math.log(THINNEST)


@wp.func
def opacity(ratio: wp.float32) -> wp.float32:
    """How solid a box is, from its energy over the start's mean of that kind."""
    depth = wp.float32(0.0)
    if wp.isfinite(ratio) and ratio > wp.float32(0.0):
        depth = wp.clamp((wp.log(ratio) - wp.float32(_LOG_THINNEST)) / wp.float32(_LOG_SPAN),
                         wp.float32(0.0), wp.float32(1.0))
    elif wp.isinf(ratio) and ratio > wp.float32(0.0):
        depth = wp.float32(1.0)
    return wp.exp(wp.float32(_LOG_FAINTEST) * (wp.float32(1.0) - depth))
