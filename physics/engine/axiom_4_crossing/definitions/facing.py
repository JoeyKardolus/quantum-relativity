"""Facing: which box of each of a joint's two volumes a row of the joint lies between.

A joint's rows run along its axis from the volume before it to the one after. A row faces
the last box along the axis of the same row of the first volume and the first box of that
row of the second.
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.places import stride_of


@wp.func
def faced(row: wp.int32, across: wp.int32, axis: wp.int32) -> wp.vec2i:
    """The box of the volume before and of the one after that the row from box `row` on faces."""
    return wp.vec2i(row + (across - 1) * stride_of(across, axis), row)
