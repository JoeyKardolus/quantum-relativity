"""Mouth: the boxes of a volume's block that look out toward one neighbour.

Contract: two Warp funcs. A block has 152 boxes on its outside. Each looks out in
exactly one of the 26 directions: step -1 along an axis where its coordinate is 0, +1
where it is 5, 0 elsewhere. The boxes looking one way are the cell's mouth toward the
neighbour there. This is book-keeping inside the cell, not a place in space: a mouth
only says which of the cell's own fluid stands ready to trade along a way.
  mouth_size(code) -> int32     16 for a face direction, 4 for an edge, 1 for a corner,
                                0 for the centre code 13
  mouth_box(code, slot) -> vec3i  the slot-th box (x, y, z) of that mouth, the boxes in
                                rising order of their free coordinates (each 1 to 4),
                                which is rising order of `x * 36 + y * 6 + z`
Two mouths that face each other across a joint have the same number of boxes, and the
n-th box of one faces the n-th box of the other: their free coordinates are the same.
That is how `cross` (logic) pairs them. A box never changes cell, and no box belongs to
two cells: a pair only says which fluid of one cell trades with which fluid of the other.
The block is where a cell's fluid is worked out, and a direction of the fluid has to
mean a neighbour, so each relation is given the boxes that look its way. That gives a
relation 16, 4 or 1 boxes: see "Open" in axiom 3's CLAUDE.md.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_3_structure.definitions.direction_codes import step_of

wp.set_module_options({"fuse_fp": False})


@wp.func
def _free_axes(code: wp.int32) -> wp.int32:
    """How many axes a direction has no step along."""
    step = step_of(code)
    free = 0
    for axis in range(3):
        if step[axis] == 0:
            free += 1
    return free


@wp.func
def mouth_size(code: wp.int32) -> wp.int32:
    """How many boxes the mouth toward `code` has."""
    free = _free_axes(code)
    if free == 2:
        return 16
    if free == 1:
        return 4
    if free == 0:
        return 1
    return 0


@wp.func
def mouth_box(code: wp.int32, slot: wp.int32) -> wp.vec3i:
    """The slot-th box of the mouth toward `code`."""
    step = step_of(code)
    left = _free_axes(code)
    box = wp.vec3i(0, 0, 0)
    for axis in range(3):
        if step[axis] > 0:
            box[axis] = 5
        elif step[axis] == 0:
            left -= 1
            if left == 1:
                box[axis] = 1 + slot // 4
            else:
                box[axis] = 1 + slot % 4
    return box
