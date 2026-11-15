"""Light code: a point's light as one byte on a log scale, so faint light survives a uint8 colour.

Contract: `light_code(light)` is a Warp function returning a float32 in [0, 255]
(round it to a uint8): 0 for light at or below DIMMEST (or not finite and not
+inf), else 1 + 254 x (log light - log DIMMEST) / (log BRIGHTEST - log DIMMEST),
light above BRIGHTEST coded as BRIGHTEST. The window decodes it (`point_cloud.py`)
as DIMMEST x (BRIGHTEST / DIMMEST) ^ ((code - 1) / 254) for a code above 0, a
relative step of about 8% per code. Needed because an average box's light at
side 256 is about 1e-4, which a plain uint8 alpha rounds to 0; light above 1
comes from a sampled point standing in for the boxes not drawn (`box_points.py`).
"""
from __future__ import annotations

import math

import warp as wp

DIMMEST = 1e-7
BRIGHTEST = 1e3

LOG_DIMMEST = math.log(DIMMEST)
LOG_SPAN = math.log(BRIGHTEST) - math.log(DIMMEST)


@wp.func
def light_code(light: wp.float32) -> wp.float32:
    """The byte (as a float to round) that carries a point's light."""
    code = wp.float32(0.0)
    if wp.isinf(light) and light > wp.float32(0.0):
        code = wp.float32(255.0)
    elif wp.isfinite(light) and light > wp.float32(DIMMEST):
        place = (wp.log(light) - wp.float32(LOG_DIMMEST)) / wp.float32(LOG_SPAN)
        code = wp.float32(1.0) + wp.float32(254.0) * wp.min(place, wp.float32(1.0))
    return code
