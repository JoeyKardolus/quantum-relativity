"""Box potential (Warp func): the energy per quantum that turns a box's fluid phase.

Contract: `box_potential(quanta, pull, push)` returns, per quantum:
  plus box   pull: an energy volume only pulls (owner, 2026-11-16: no size at rest, no
             push of its own); the pull is felt by plus quanta only
  minus box  push: its joint's push (cell_laws.py, push * (wanted / size) ** steepness,
             one value for the whole joint); a joint carries the pull but does not feel it
  empty box  0: empty space does nothing
The fluid's phase turns by it (on_site.py).
"""
from __future__ import annotations

import warp as wp


@wp.func
def box_potential(quanta: int, pull: float, push: float) -> float:
    """The energy per quantum a box's fluid feels: the pull for plus, its joint's push for minus, 0 for empty."""
    if quanta > 0:
        return pull
    if quanta < 0:
        return push
    return 0.0
