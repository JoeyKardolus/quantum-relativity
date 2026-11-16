"""Site turn (Warp func): one box's fluid phase after turning by the energy per quantum it feels.

Contract: `site_turn(amount, pull, push, packing, pressure, angle, span, rate)` returns
the phase `angle` turned by -fluid_potential(amount, pull, push, packing, pressure) *
span * rate (fluid_potential.py; `rate` the cell's clock rate), brought into [-pi, pi).
The on-site half of the fluid step, shared by on_site.py (every box at once) and
later_substeps.py (cell by cell), so both turn a box by exactly the same number.
"""
from __future__ import annotations

import warp as wp

from physics.engine.fluid_potential import fluid_potential

PI = 3.14159265358979


@wp.func
def site_turn(amount: int, pull: float, push: float, packing: float, pressure: float, angle: float, span: float,
              rate: float) -> float:
    """The phase after an exact turn by the box's potential for `span` of its cell's clock."""
    energy = fluid_potential(amount, pull, push, packing, pressure)
    turned = angle - energy * span * rate
    return turned - 2.0 * PI * wp.floor((turned + PI) / (2.0 * PI))
