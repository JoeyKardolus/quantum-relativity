"""Cells: the grid as host arrays, one row per cell id.

Contract: `cells(state)` returns a dict of numpy arrays over ids 0 .. state.count - 1
(copies; the state is not changed):
  kind     1 energy volume, -1 joint, 0 no cell (an id a collapse emptied)
  quanta   the cell's quanta ignoring sign (int64)
  length   its size by the size law; packing, rate, clock, pull, squeeze, jiggle as in
           state.py
  place    (count, 3) the lattice indices of a lattice id, -1 for a cell a tear made
A cell has no geometry: a viewer lays the cells out from their relations
(state.neighbours) and lengths.
"""
from __future__ import annotations

import numpy as np

from physics.engine.state import State


def cells(state: State) -> dict:
    """Host copies of every per-cell number, by cell id."""
    side = state.side
    ids = np.arange(state.count)
    place = np.stack([ids // (side * side), (ids // side) % side, ids % side], axis=1)
    place[ids >= state.lattice] = -1
    found = {name: state.host(name) for name in ("length", "packing", "rate", "clock", "pull", "squeeze", "jiggle")}
    found.update(kind=state.kind_h[:state.count].copy(), quanta=state.host("cell_quanta"), place=place)
    return found
