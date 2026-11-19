"""Facing: which box of each of a joint's two volumes a row of the joint lies between.

A joint's rows run along its axis from the volume before it to the one after. A row faces
the last box along the axis of a row of the first volume and the first box of a row of the
second, each in the joint's part of that side (`relations.py`). Where a part holds fewer
boxes than the joint has rows, several rows face one box: energy then crosses along the
first of them only (`leads`), so no box is in two crossings of one launch.
"""
import warp as wp

from physics.engine.axiom_3_structure.definitions.relations import box_along
from physics.engine.axiom_4_crossing.definitions.passage import row_of, stride_of


@wp.func
def faced(joint_part: wp.array2d(dtype=wp.vec4), joint: wp.int32, first: wp.int32,
          third: wp.int32, across: wp.int32, axis: wp.int32) -> wp.vec2i:
    """The box of the volume before and of the one after that a joint's row faces; -1: none."""
    found = wp.vec2i(-1, -1)
    for end in range(2):
        part = joint_part[joint, end]
        one = box_along(part[0], part[1], first, across)
        two = box_along(part[2], part[3], third, across)
        if one >= 0 and two >= 0:
            found[end] = (row_of(one, two, across, axis)
                          + (1 - end) * (across - 1) * stride_of(across, axis))
    return found


@wp.func
def leads(joint_part: wp.array2d(dtype=wp.vec4), joint: wp.int32, first: wp.int32,
          third: wp.int32, across: wp.int32) -> wp.int32:
    """1 when the row is the first of the joint's rows to face its box, in both volumes."""
    alone = wp.int32(1)
    for end in range(2):
        part = joint_part[joint, end]
        if first > 0 and (box_along(part[0], part[1], first, across)
                          == box_along(part[0], part[1], first - 1, across)):
            alone = 0
        if third > 0 and (box_along(part[2], part[3], third, across)
                          == box_along(part[2], part[3], third - 1, across)):
            alone = 0
    return alone


@wp.func
def row_facing(start: float, wide: float, box: wp.int32, across: wp.int32) -> wp.int32:
    """The first of a joint's rows that faces box `box` along one direction of a part; -1: none."""
    for row in range(across):
        if box_along(start, wide, row, across) == box:
            return row
    return -1
