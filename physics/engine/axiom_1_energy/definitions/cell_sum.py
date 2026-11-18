"""Cell sum: the quanta one cell holds, counted from its own boxes.

Contract: `cell_sum(quanta, cell)` (Warp func) returns the sum of the cell's 216 boxes
ignoring sign, as a whole int64, so the count is exact. It reads the boxes of that one
cell and nothing else.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import BLOCK

wp.set_module_options({"fuse_fp": False})

_BLOCK = wp.constant(BLOCK)


@wp.func
def cell_sum(quanta: wp.array4d(dtype=wp.int32), cell: wp.int32) -> wp.int64:
    """The quanta in one cell's block, ignoring sign."""
    held = wp.int64(0)
    for across in range(_BLOCK):
        for along in range(_BLOCK):
            for up in range(_BLOCK):
                held += wp.int64(wp.abs(quanta[cell, across, along, up]))
    return held
