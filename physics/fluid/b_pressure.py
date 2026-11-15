"""B pressure: how hard the pushing energy in a box pushes.

Contract: a Warp function, usable inside kernels only.
`b_pressure(b, b_rest, push, power, collapse)`: with compression c = min(b / b_rest,
collapse), P = push * b_rest * c**power / power. It rises steeply with compression
(power 3 by default) and stops rising past the collapse limit, so a core packed past
it has no push left to resist the pull: the runaway. B always pushes a little, also
below rest, so the edge of the energy has nothing holding it in from outside and
spreads outward (open space, drift).
"""
from __future__ import annotations

import warp as wp


@wp.func
def b_pressure(b: float, b_rest: float, push: float, power: float, collapse: float) -> float:
    """The push of B-energy b per box."""
    compression = wp.min(wp.max(b, 0.0) / b_rest, collapse)
    return push * b_rest * wp.pow(compression, power) / power

