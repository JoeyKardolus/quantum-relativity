"""Fluid potential (Warp func): the energy per quantum a box's fluid feels, its cell's pull plus the fluid's own pressure.

Contract: `fluid_potential(quanta, pull, packing, pressure)` returns `pull` (the cell's
pull for a plus box; the caller passes 0 for a joint's box, which feels no pull) plus
the fluid's own pressure inside its cell, pressure * (|quanta| / packing - 1), packing
the cell's mean quanta per box and pressure settings.fluid_pressure. Owner, 2026-11-16:
a whirl holds its own core open by its spinning, as in any fluid; the energy is a
quantum fluid (Gross-Pitaevskii style), so a box denser than its cell's mean is pushed
and a thinner one drawn in. The pressure is measured from the cell's own mean, so it
moves energy only inside the cell. The healing length, the width of a whirl's core, is
1 / sqrt(2 pressure) boxes (hbar = box mass = 1): 1 box at the default. A packing of 0
feels no pressure.
"""
from __future__ import annotations

import warp as wp


@wp.func
def fluid_potential(quanta: int, pull: float, packing: float, pressure: float) -> float:
    """The pull plus the fluid's own pressure against its cell's mean per box."""
    own = float(0.0)
    if packing > 0.0:
        own = pressure * (float(wp.abs(quanta)) / packing - 1.0)
    return pull + own
