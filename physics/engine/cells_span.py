"""Cells span: the room, in boxes, that a few cells of one universe lie in, measured on the GPU.

Contract: `cells_span(state, universe, cells, margin=1)` returns (low, extent): per axis
the first coordinate in the universe and the length of the smallest box-shaped room that
holds every box of the cell ids in `cells` with `margin` boxes of room round them. In a
universe that wraps round, the room is measured the short way round the edge (its
coordinates then run low .. low + extent - 1 taken round the side); in one with walls it
stops at the walls. Returns None when the cells have no boxes, or when on some axis the
room would reach all the way round a wrapping universe (the caller then takes the whole
universe). One launch over the boxes; nothing but two rows of numbers comes to the host.
volume_split.py uses it to read and write only the torn volume's room (2026-11-17:
copying every box to the host for one tear took 17 s a step at 147 million boxes).
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.state import State


@wp.kernel
def _span(owner: wp.array3d(dtype=wp.int32), wanted: wp.array(dtype=wp.int32), first: int, side: int,
          spans: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    cell = owner[first + ti, tj, tk]
    if cell < 0 or wanted[cell] == 0:
        return
    here = wp.vec3i(ti, tj, tk)
    for axis in range(3):
        moved = (here[axis] + side // 2) % side
        wp.atomic_min(spans, 4 * axis, here[axis])
        wp.atomic_max(spans, 4 * axis + 1, here[axis])
        wp.atomic_min(spans, 4 * axis + 2, moved)
        wp.atomic_max(spans, 4 * axis + 3, moved)


def cells_span(state: State, universe: int, cells, margin: int = 1):
    """(low, extent) per axis of the room holding the cells' boxes plus a margin, or None (see the contract)."""
    device = state.quanta.device
    side = state.side
    wanted = np.zeros(state.capacity, dtype=np.int32)
    wanted[np.asarray(cells, dtype=np.int64)] = 1
    spans = wp.array(np.tile(np.array([side, -1, side, -1], dtype=np.int32), 3), dtype=wp.int32, device=device)
    wp.launch(_span, dim=(side, side, side), device=device,
              inputs=[state.owner, wp.array(wanted, dtype=wp.int32, device=device), universe * side, side, spans])
    found = spans.numpy().reshape(3, 4)
    if (found[:, 1] < 0).any():
        return None
    low = found[:, 0] - margin
    extent = found[:, 1] - found[:, 0] + 1 + 2 * margin
    if not state.wrap:
        high = np.minimum(low + extent, side)
        low = np.maximum(low, 0)
        return low, high - low
    moved = found[:, 3] - found[:, 2] + 1 + 2 * margin
    shorter = moved < extent
    low = np.where(shorter, found[:, 2] - side // 2 - margin, low) % side
    extent = np.where(shorter, moved, extent)
    if (extent >= side).any():
        return None
    return low, extent
