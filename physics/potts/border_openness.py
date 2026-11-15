"""Border openness (Warp function): how much of a flow may cross from one box into its face neighbour.

Contract: `border_openness(owner, kind, compression, seam, here, there, seal, side)` takes two
face-neighbouring boxes and returns 1 when both belong to one cell (frictionless,
nothing in the way); 0 when `there` is empty or outside the grid, between two cells of
the same kind (ABAB forbids them touching), when the joint's box of the two (the B
side) is not on a seam (seam[box] == 0, seam_flags.py; rule 6: the leak runs along the seams, never
through a joint's middle), or when that joint is squeezed to `seal` or past it (bible:
past the seal point nothing passes); else 1 for a joint at or under compression 1,
falling straight to 0 at the seal (the more squeezed the joint, the narrower its seam
and the less it lets through).
"""
from __future__ import annotations

import warp as wp

from physics.potts.box_owner import box_owner


@wp.func
def border_openness(owner: wp.array3d(dtype=wp.int32), kind: wp.array(dtype=wp.int32),
                    compression: wp.array(dtype=wp.float32), seam: wp.array3d(dtype=wp.int32), here: wp.vec3i,
                    there: wp.vec3i, seal: float,
                    side: int) -> float:
    """1 inside a cell; through a joint's seam 1 relaxed, down to 0 at the seal; 0 elsewhere."""
    sender = box_owner(owner, here[0], here[1], here[2], side)
    receiver = box_owner(owner, there[0], there[1], there[2], side)
    if sender < 0 or receiver < 0:
        return 0.0
    if sender == receiver:
        return 1.0
    if kind[sender] == kind[receiver]:
        return 0.0
    joint = receiver
    joint_box = there
    if kind[sender] == 1:
        joint = sender
        joint_box = here
    if seam[joint_box[0], joint_box[1], joint_box[2]] == 0:
        return 0.0
    squeeze = compression[joint]
    if squeeze >= seal:
        return 0.0
    if squeeze <= 1.0:
        return 1.0
    return (seal - squeeze) / (seal - 1.0)
