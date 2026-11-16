"""Openness (Warp func): how freely the fluid's hops cross one face between two Planck boxes.

Contract: `openness(near, far)` takes the two boxes' signed quanta and returns
  1  between two boxes of one sign (inside an energy volume, or inside the joint web:
     frictionless, bible rule 4)
  0  when either box is empty (zero is no space) or the signs differ: a hop never moves
     quanta across a plus-minus face, because quanta never take the other kind's sign
     (bible rule 4: plus and minus never cancel, only joint collapse turns minus into
     plus). Energy passes from energy volume to energy volume through a joint as one
     transfer across it (leak.py), graded there by the joint's squeeze (leak_openness.py).
"""
from __future__ import annotations

import warp as wp


@wp.func
def openness(near: int, far: int) -> float:
    """1 between two occupied boxes of one sign, else 0."""
    if near == 0 or far == 0:
        return 0.0
    if (near > 0) == (far > 0):
        return 1.0
    return 0.0
