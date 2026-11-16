"""Box energy (Warp func): the energy a box's quanta hold in the pull and in their own push.

Contract: `box_energy(quanta, pull, volume_rest, volume_stiffness, joint_rest, push,
push_steepness)` returns
  plus box   quanta * pull + volume_stiffness * quanta**2 / (2 volume_rest)
  minus box  push * joint_rest * (|quanta| / joint_rest) ** (push_steepness + 1) / (push_steepness + 1)
  empty box  0
Its derivative by the quanta is box_potential.py. Claims (claim.py) take a box move only
when it lowers the sum of this over the boxes it touches. A bookkeeping energy for moves,
not the conserved amount: the conserved thing is the quanta themselves.
"""
from __future__ import annotations

import warp as wp


@wp.func
def box_energy(quanta: int, pull: float, volume_rest: float, volume_stiffness: float, joint_rest: float,
               push: float, push_steepness: float) -> float:
    """The pull and push energy of `quanta` in one box."""
    if quanta > 0:
        amount = float(quanta)
        return amount * pull + volume_stiffness * amount * amount / (2.0 * volume_rest)
    if quanta < 0:
        power = push_steepness + 1.0
        return push * joint_rest * wp.pow(float(-quanta) / joint_rest, power) / power
    return 0.0
