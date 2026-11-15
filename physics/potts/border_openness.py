"""Border openness (Warp function): how freely the energy fluid may hop from one box into its face neighbour.

Contract: `border_openness(owner, kind, compression, seam, here, there, seal, side)` takes two
face-neighbouring boxes and returns 1 when both belong to one cell (frictionless,
nothing in the way); 0 when either is empty or outside the grid, between two energy
volumes (they never touch: each is wrapped in joints), or when a joint's box of the two
is not on a seam (seam[box] == 0, seam_flags.py; rule 6: the leak runs along the
seams, never through a joint's middle). Otherwise, an energy volume against a joint or
two joints meeting in the web, the most squeezed joint of the two, at compression c,
sets it: 0 at or past `seal` (bible: past the seal point nothing passes), 1 at or
under c = 1, falling straight to 0 between (the more squeezed, the narrower the seam).
The fluid's hop across that face is scaled by this number (hop.py).
"""
from __future__ import annotations

import warp as wp

from physics.potts.box_owner import box_owner


@wp.func
def _joint_openness(compression: wp.array(dtype=wp.float32), seam: wp.array3d(dtype=wp.int32), joint: int,
                    box: wp.vec3i, seal: float) -> float:
    """1 for a relaxed joint's seam box, down to 0 at the seal; 0 off the seams."""
    if seam[box[0], box[1], box[2]] == 0:
        return 0.0
    squeeze = compression[joint]
    if squeeze >= seal:
        return 0.0
    if squeeze <= 1.0:
        return 1.0
    return (seal - squeeze) / (seal - 1.0)


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
    if kind[sender] == 0 and kind[receiver] == 0:
        return 0.0
    openness = float(1.0)
    if kind[sender] == 1:
        openness = wp.min(openness, _joint_openness(compression, seam, sender, here, seal))
    if kind[receiver] == 1:
        openness = wp.min(openness, _joint_openness(compression, seam, receiver, there, seal))
    return openness
