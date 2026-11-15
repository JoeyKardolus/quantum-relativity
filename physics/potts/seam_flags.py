"""Seam flags: mark every joint box that lies on a seam, once per step, for the fluid substeps.

Contract: `seam_flags(universe)` writes universe.seam over the active region: 1 for a
box seam_box.py calls a seam box (a joint's box sharing an edge with another joint's
box), else 0. Box owners do not change during the fluid substeps, so the flags hold
for all of them and border_openness.py reads them instead of looking at twelve edge
neighbours per face. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.potts.seam_box import seam_box
from physics.potts.universe import Universe


@wp.kernel
def _flags(owner: wp.array3d(dtype=wp.int32), kind: wp.array(dtype=wp.int32), low: wp.vec3i, side: int,
           seam: wp.array3d(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    i = ti + low[0]
    j = tj + low[1]
    k = tk + low[2]
    seam[i, j, k] = seam_box(owner, kind, i, j, k, side)


def seam_flags(universe: Universe) -> None:
    """Flag the seam boxes of the active region."""
    dims = tuple(top - bottom for bottom, top in zip(universe.low, universe.high))
    wp.launch(_flags, dim=dims, device=universe.owner.device,
              inputs=[universe.owner, universe.kind, wp.vec3i(*universe.low), universe.side, universe.seam])
