"""Host boxes: one universe's per-box arrays copied from the GPU to numpy.

Contract: `host_boxes(state, universe=0)` returns a dict of numpy arrays of shape
(side, side, side), indexed [x, y, z] in that universe's own coordinates: quanta
(int32, signed whole numbers), phase, clock and pull (float32), owner (int32, the
extended-lattice cell each box belongs to, numbered over the universe's own lattice
(nx + 2, ny + 2, nz + 2) as lattice_kinds and State.from_host take it), plus "cells"
(the universe's real cells per axis, a tuple of three ints, (0, 0, 0) without a cell
lattice) and "corners" (float32 (nx + 3, ny + 3, nz + 3, 3): the lattice vertices in
box units). Copies once per call; the state is not changed. Raises ValueError for a
universe the batch does not hold.
"""
from __future__ import annotations

import numpy as np

from physics.engine.state import State


def host_boxes(state: State, universe: int = 0) -> dict:
    """quanta, phase, clock, pull and owner of one universe as host arrays, with its lattice."""
    rows = state.universe(universe)
    boxes = {name: np.ascontiguousarray(getattr(state, name)[rows.start:rows.stop].numpy())
             for name in ("quanta", "phase", "clock", "pull", "owner")}
    extent = state.extent
    vertices = tuple(count + 1 for count in extent)
    corners = state.corners.numpy().reshape((state.universes,) + vertices + (3,))[universe]
    counts = tuple(int(count) for count in state.cells[universe])
    boxes["cells"] = counts
    if not any(counts):
        boxes["corners"] = corners.copy()
        return boxes
    own = tuple(count + 2 for count in counts)
    boxes["corners"] = np.ascontiguousarray(corners[:own[0] + 1, :own[1] + 1, :own[2] + 1])
    if own != tuple(extent):
        place = np.unravel_index(boxes["owner"], extent)
        boxes["owner"] = np.ravel_multi_index(place, own).astype(np.int32)
    return boxes
