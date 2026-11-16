"""Box potential (Warp func): the energy per quantum that turns a box's fluid phase.

Contract: `box_potential(quanta, pull, volume_rest, volume_stiffness, joint_rest, push,
push_steepness)` returns, per quantum:
  plus box   pull + volume_stiffness * quanta / volume_rest: the pull (felt by plus
             quanta only) and the energy volume's own fluid push (Gross-Pitaevskii g n)
  minus box  push * (|quanta| / joint_rest) ** push_steepness: the joint's push, rising
             steeply as it is packed (a hard wall); joints carry the pull but do not feel it
  empty box  0
The fluid's phase turns by it (on_site.py); the cell corners follow their own rule,
size_energy / quanta (move_corners.py).
"""
from __future__ import annotations

import warp as wp


@wp.func
def box_potential(quanta: int, pull: float, volume_rest: float, volume_stiffness: float, joint_rest: float,
                  push: float, push_steepness: float) -> float:
    """The energy per quantum a box's fluid feels: pull and own push for plus, the steep joint push for minus."""
    if quanta > 0:
        return pull + volume_stiffness * float(quanta) / volume_rest
    if quanta < 0:
        return push * wp.pow(float(-quanta) / joint_rest, push_steepness)
    return 0.0
