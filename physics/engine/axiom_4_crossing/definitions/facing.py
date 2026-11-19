"""Facing: which boxes meet where a joint lies between two volumes.

A joint lies along one direction, between the volume before it and the one after. Each
cell's ring for that direction has a cap that faces on and a cap that faces back
(`ring_boxes.py`). Row for row, the front cap of the volume before meets the joint's back
cap, and the joint's front cap meets the back cap of the volume after. A joint turns the
other way, so its fluid moves with each volume's where they touch: its two caps hold the
very phases the two volumes' caps hold. Two volumes with no joint between would meet half
a turn apart and grind.
"""
import warp as wp

from physics.engine.axiom_2_swirls.definitions.ring_boxes import cap_box


@wp.func
def faced(row: wp.int32, across: wp.int32, axis: wp.int32) -> wp.vec2i:
    """The box of the volume before and of the one after that the joint's row `row` lies between.

    The joint's own two boxes of that row are the same two numbers the other way round.
    """
    return wp.vec2i(cap_box(across, axis, row, 1), cap_box(across, axis, row, 0))

