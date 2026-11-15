"""Seam box (Warp function): whether a joint's box lies on one of its seams.

Contract: `seam_box(owner, kind, i, j, k, side)` returns 1 when box (i, j, k) belongs to
a joint and one of its 26 neighbours (sharing a face, an edge or a corner with it)
belongs to another joint, else 0. Joints touch each other and form one web, and the
seams are where they meet (bible foundation 1, owner 2026-11-16); rule 6: energy
leaks along the seams, never through a joint's middle, so only a seam box lets energy
in or out of its joint (border_openness.py).
"""
from __future__ import annotations

import warp as wp

from physics.potts.box_owner import box_owner


@wp.func
def seam_box(owner: wp.array3d(dtype=wp.int32), kind: wp.array(dtype=wp.int32), i: int, j: int, k: int,
             side: int) -> int:
    """1 when the joint box (i, j, k) has another joint's box among its 26 neighbours."""
    cell = box_owner(owner, i, j, k, side)
    if cell < 0:
        return 0
    if kind[cell] != 1:
        return 0
    for near in range(27):
        di = near // 9 - 1
        dj = (near // 3) % 3 - 1
        dk = near % 3 - 1
        other = box_owner(owner, i + di, j + dj, k + dk, side)
        if other >= 0 and other != cell:
            if kind[other] == 1:
                return 1
    return 0
