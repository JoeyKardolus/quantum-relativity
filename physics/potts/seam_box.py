"""Seam box (Warp function): whether a joint's box lies on one of its seams.

Contract: `seam_box(owner, kind, i, j, k, side)` returns 1 when box (i, j, k) belongs to
a joint and one of its twelve edge neighbours (the boxes sharing only an edge with it)
belongs to another joint, else 0. Joints touch each other only along such thin edges,
the seams (bible rule 2); rule 6: energy leaks along the seams, never through a joint's
middle, so only a seam box lets energy in or out of its joint (border_openness.py).
"""
from __future__ import annotations

import warp as wp

from physics.potts.box_owner import box_owner


@wp.func
def seam_box(owner: wp.array3d(dtype=wp.int32), kind: wp.array(dtype=wp.int32), i: int, j: int, k: int,
             side: int) -> int:
    """1 when the joint box (i, j, k) shares an edge with another joint's box."""
    cell = box_owner(owner, i, j, k, side)
    if cell < 0:
        return 0
    if kind[cell] != 1:
        return 0
    for pair in range(12):
        plane = pair // 4
        first = 1 - 2 * (pair % 2)
        second = 1 - 2 * ((pair // 2) % 2)
        di = 0
        dj = 0
        dk = 0
        if plane == 0:
            di = first
            dj = second
        elif plane == 1:
            dj = first
            dk = second
        else:
            di = first
            dk = second
        other = box_owner(owner, i + di, j + dj, k + dk, side)
        if other >= 0 and other != cell:
            if kind[other] == 1:
                return 1
    return 0
