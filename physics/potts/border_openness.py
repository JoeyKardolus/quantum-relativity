"""Border openness (Warp function): how much of a flow may cross from one cell into its neighbour.

Contract: `border_openness(kind, compression, sender, receiver, seal)` returns 1 inside
one cell (sender == receiver: frictionless, nothing in the way); 0 between two cells
of the same kind (ABAB forbids them touching, so nothing passes) or when the joint on
the border (the B of the two) is squeezed to `seal` or past it (bible: past the seal
point nothing passes); else 1 for a joint at or under compression 1, falling straight
to 0 at the seal (the more squeezed the joint, the less it lets through).
"""
from __future__ import annotations

import warp as wp


@wp.func
def border_openness(kind: wp.array(dtype=wp.int32), compression: wp.array(dtype=wp.float32), sender: int,
                    receiver: int, seal: float) -> float:
    """1 inside a cell; through a joint 1 relaxed, down to 0 at the seal; 0 between like kinds."""
    if sender == receiver:
        return 1.0
    if kind[sender] == kind[receiver]:
        return 0.0
    joint = receiver
    if kind[sender] == 1:
        joint = sender
    squeeze = compression[joint]
    if squeeze >= seal:
        return 0.0
    if squeeze <= 1.0:
        return 1.0
    return (seal - squeeze) / (seal - 1.0)
