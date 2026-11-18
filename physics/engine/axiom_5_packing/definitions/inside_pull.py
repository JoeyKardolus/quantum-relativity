"""Inside pull: where a box of a cell stands between the cell's middle and its sides.

Contract: two Warp funcs for one box (across, along, up) of a block and one `corner`, a number 0
to 7 whose bits say along which of the three axes the corner lies at the cell's side.
A box stands, along each axis, a share `w = abs(coordinate - 2.5) / 2.5` of the way from
the middle of the cell (0) to its side (1), toward the side its coordinate is nearer to.
  corner_code(across, along, up, corner) -> int32
      the direction code (direction_codes.py) of that corner: along each axis whose bit
      is set, the step toward the side the box is nearer to; 0 along the others. Corner
      0 is the centre code 13, the middle of the cell
  corner_weight(across, along, up, corner) -> float64
      how much the pull at that corner counts for the box: the product over the three
      axes of `w` where the bit is set and `1 - w` where it is not. The eight weights
      sum to 1
So the pull a box feels is the pull at the cell's middle and at its sides, taken in
proportion to where the box stands between them: the pull slopes evenly across the
whole cell and is no step at its skin. This is book-keeping inside the cell (a
direction of the fluid has to mean a neighbour); it gives no box a place in space.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import BLOCK

wp.set_module_options({"fuse_fp": False})

_MIDDLE = wp.constant(wp.float64((BLOCK - 1) / 2.0))
_CODE_STEPS = wp.constant(wp.vec3i(9, 3, 1))


@wp.func
def _toward_side(coordinate: wp.int32) -> wp.float64:
    """How far a coordinate stands from the middle of the block toward its side, signed, -1 to 1."""
    return (wp.float64(coordinate) - _MIDDLE) / _MIDDLE


@wp.func
def corner_code(across: wp.int32, along: wp.int32, up: wp.int32, corner: wp.int32) -> wp.int32:
    """The direction code of one of the eight corners a box stands between."""
    box = wp.vec3i(across, along, up)
    code = 0
    for axis in range(3):
        step = 0
        if (corner >> axis) & 1 == 1:
            step = 1
            if _toward_side(box[axis]) < wp.float64(0.0):
                step = -1
        code += (step + 1) * _CODE_STEPS[axis]
    return code


@wp.func
def corner_weight(across: wp.int32, along: wp.int32, up: wp.int32, corner: wp.int32) -> wp.float64:
    """How much the pull at one of the box's eight corners counts for the box."""
    box = wp.vec3i(across, along, up)
    weight = wp.float64(1.0)
    for axis in range(3):
        share = wp.abs(_toward_side(box[axis]))
        if (corner >> axis) & 1 == 1:
            weight *= share
        else:
            weight *= wp.float64(1.0) - share
    return weight
