"""Openness (Warp func): how freely the fluid crosses one face between two Planck boxes.

Contract: `openness(near, far, joint_rest, seal_squeeze)` takes the two boxes' signed
quanta and returns a number in [0, 1]:
  0  when either box is empty (zero is no space: nothing flows into or out of it)
  1  between two boxes of one sign (inside an energy volume, or inside the joint web:
     frictionless, bible rule 4)
  1 - s / seal_squeeze, floored at 0, on a crossing face (one plus box, one minus box),
     s the minus box's squeeze when positive (squeeze.py): the more squeezed the joint,
     the less passes; past the seal nothing passes, so packed lumps seal themselves.
Every plus-minus face is a crossing face: the signed state holds no joint labels, so
"where two joints touch" (the bible's seam) cannot be read from it; this engine lets
energy cross where the kinds meet, graded by the joint box's squeeze.
"""
from __future__ import annotations

import warp as wp

from physics.engine.squeeze import squeeze


@wp.func
def openness(near: int, far: int, joint_rest: float, seal_squeeze: float) -> float:
    """1 inside one kind, graded by the joint's squeeze across kinds, 0 into empty space or past the seal."""
    if near == 0 or far == 0:
        return 0.0
    if (near > 0) == (far > 0):
        return 1.0
    pressed = wp.max(squeeze(wp.min(near, far), joint_rest), 0.0)
    return wp.max(1.0 - pressed / seal_squeeze, 0.0)
