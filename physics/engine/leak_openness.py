"""Leak openness (Warp func): how freely energy leaks through a joint between two energy volume boxes.

Contract: `leak_openness(packing, joint_rest, seal_squeeze)` takes the joint's packing
where the leak crosses it (mean quanta per joint box along the crossing, leak.py) and
returns 1 - s / seal_squeeze floored at 0, s = packing / joint_rest - 1 when positive:
an unsqueezed or slack joint lets through fully, the more squeezed the less, and past
the seal point nothing passes, so packed lumps seal themselves (bible rule 4 and 6). A
joint's thickness in Planck boxes does not enter: boxes are only the computing grid,
and energy crosses exactly one joint cell however many boxes thick it is (owner,
2026-11-16). Returns 0 for a packing of 0 or less (no joint).
"""
from __future__ import annotations

import warp as wp


@wp.func
def leak_openness(packing: float, joint_rest: float, seal_squeeze: float) -> float:
    """1 for a slack joint, falling with its squeeze to 0 at the seal; 0 where there is no joint."""
    if packing <= 0.0:
        return 0.0
    pressed = wp.max(packing / joint_rest - 1.0, 0.0)
    return wp.max(1.0 - pressed / seal_squeeze, 0.0)
