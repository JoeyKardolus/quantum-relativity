"""B wave speed: how fast a change of push travels through the pushing energy in a box.

Contract: a Warp function, usable inside kernels only.
`b_wave_speed(b, b_rest, push, power, collapse)`: sqrt(dP/db) = sqrt(push *
c**(power - 1)), c = b / b_rest, below the collapse limit (b_pressure.py), 0 past it
(no rising pressure, no wave).
"""
from __future__ import annotations

import warp as wp


@wp.func
def b_wave_speed(b: float, b_rest: float, push: float, power: float, collapse: float) -> float:
    """How fast a change of push travels through B-energy b per box."""
    compression = wp.max(b, 0.0) / b_rest
    if compression >= collapse:
        return 0.0
    return wp.sqrt(push * wp.pow(compression, power - 1.0))
