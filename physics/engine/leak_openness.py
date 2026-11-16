"""Leak openness (Warp func): how freely energy leaks across one joint box between two energy volume boxes.

Contract: `leak_openness(middle, joint_rest, seal_squeeze)` takes the signed quanta of
the joint box a leak crosses and returns 1 - s / seal_squeeze floored at 0, s its
squeeze when positive (squeeze.py): an unsqueezed or slack joint lets through fully,
the more squeezed the less, and past the seal point nothing passes, so packed lumps
seal themselves (bible rule 4 and 6). Returns 0 when `middle` is not a joint box.
"""
from __future__ import annotations

import warp as wp

from physics.engine.squeeze import squeeze


@wp.func
def leak_openness(middle: int, joint_rest: float, seal_squeeze: float) -> float:
    """1 for a slack joint box, falling with its squeeze to 0 at the seal; 0 for a box that is not a joint's."""
    if middle >= 0:
        return 0.0
    pressed = wp.max(squeeze(middle, joint_rest), 0.0)
    return wp.max(1.0 - pressed / seal_squeeze, 0.0)
