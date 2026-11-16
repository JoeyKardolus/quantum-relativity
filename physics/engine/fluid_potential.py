"""Fluid potential (Warp func): the energy per quantum a box's fluid feels, its cell's pull or push plus the fluid's own pressure.

Contract: `fluid_potential(quanta, pull, push, packing, pressure)` returns
box_potential(quanta, pull, push) (box_potential.py: the pull for plus, the joint's push
for minus, 0 for empty) plus the fluid's own pressure inside its cell,
pressure * (|quanta| / packing - 1), packing the cell's mean quanta per box (cell_count.py)
and pressure settings.push. Owner, 2026-11-16: a whirl holds its own core open by its
spinning, as in any fluid; the energy is a quantum fluid (Gross-Pitaevskii style), so a
box denser than its cell's mean is pushed and a thinner one drawn in. The pressure is
measured from the cell's own mean, so it moves energy only inside the cell: a box at
its cell's mean feels nothing of it and a cell as a whole gains no push (a volume only
pulls). The healing length, the width of a whirl's core, is 1 / sqrt(2 pressure)
boxes (hbar = box mass = 1): 1 box at the default push. Empty boxes and a packing of 0
feel 0.
"""
from __future__ import annotations

import warp as wp

from physics.engine.box_potential import box_potential


@wp.func
def fluid_potential(quanta: int, pull: float, push: float, packing: float, pressure: float) -> float:
    """box_potential plus the fluid's own pressure against its cell's mean packing."""
    if quanta == 0:
        return 0.0
    own = float(0.0)
    if packing > 0.0:
        own = pressure * (float(wp.abs(quanta)) / packing - 1.0)
    return box_potential(quanta, pull, push) + own
