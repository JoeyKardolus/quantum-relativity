"""Active region: the box bounds the box kernels run over, the occupied boxes widened by PAD.

Contract: `active_region(universe)` finds the occupied boxes' bounds on the GPU, looking
only inside the current region (box moves claim boxes only inside it, box_moves.py, so
no occupied box lies outside), and sets universe.low and
universe.high (host int triples, high exclusive) to those bounds widened by PAD and
clipped to the grid. Space beyond is empty and nothing there changes, so a kernel that
skips it skips nothing. One small copy to the host. Returns None.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.potts.universe import PAD, Universe


@wp.kernel
def _bounds(owner: wp.array3d(dtype=wp.int32), low: wp.vec3i, out: wp.array(dtype=wp.int32)):
    i, j, k = wp.tid()
    here = wp.vec3i(i + low[0], j + low[1], k + low[2])
    if owner[here[0], here[1], here[2]] < 0:
        return
    for axis in range(3):
        wp.atomic_min(out, axis, here[axis])
        wp.atomic_max(out, 3 + axis, here[axis])


def active_region(universe: Universe) -> None:
    """Refresh universe.low / high: the occupied bounds widened by PAD, clipped to the grid."""
    side = universe.side
    out = wp.array(np.array([side, side, side, -1, -1, -1], dtype=np.int32), dtype=wp.int32,
                   device=universe.owner.device)
    dims = tuple(top - bottom for bottom, top in zip(universe.low, universe.high))
    wp.launch(_bounds, dim=dims, device=universe.owner.device,
              inputs=[universe.owner, wp.vec3i(*universe.low), out])
    found = out.numpy()
    if found[3] < 0:
        return
    universe.low = tuple(int(max(0, value - PAD)) for value in found[:3])
    universe.high = tuple(int(min(side, value + 1 + PAD)) for value in found[3:])
