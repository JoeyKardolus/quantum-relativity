"""Squeeze (Warp func): how far a joint box is pressed beyond its rest packing.

Contract: `squeeze(quanta, joint_rest)` returns |quanta| / joint_rest - 1 for a minus
box (positive squeezed, negative slack) and 0 for a plus or empty box. The seal
(openness.py) and the collapse (collapse.py) read it; the push (box_potential.py) rises
steeply with the same packing.
"""
from __future__ import annotations

import warp as wp


@wp.func
def squeeze(quanta: int, joint_rest: float) -> float:
    """A joint box's packing over its rest packing, minus 1; 0 when the box is not a joint's."""
    if quanta >= 0:
        return 0.0
    return float(-quanta) / joint_rest - 1.0
