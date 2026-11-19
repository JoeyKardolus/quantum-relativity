"""Passage: how freely plus energy passes a joint, read off the joint's own fluid.

A joint stands between two volumes along one direction, its axis. `row_minus` is the minus
quanta, a box, of the joint's outside boxes once round its core at one row. Crossing energy
rubs against them: `rub_share` is the minus share of all the energy met in the passage.
The rubbing share is held back and stays in the volume it came from (`cross.py`).
No number is set here: the joint's quanta decide.
"""
import warp as wp

from physics.engine.axiom_2_swirls.definitions.ring_boxes import box_at


@wp.func
def row_minus(box_quanta: wp.array2d(dtype=wp.int32), slot: wp.int32, row: wp.int32,
              across: wp.int32, axis: wp.int32) -> wp.float64:
    """The minus quanta a box in the joint's outside boxes once round its core at row `row` (0 or less: no joint)."""
    held = wp.int32(0)
    for around in range(across):
        held -= box_quanta[slot, box_at(across, axis, across / 3 - 1, around, row % across)]
    return wp.float64(held) / wp.float64(across)


@wp.func
def rub_share(minus: wp.float64, low_has: wp.int32, high_has: wp.int32) -> wp.float64:
    """The share of crossing energy that rubs: minus over minus and plus, a box, in the passage."""
    plus = wp.float64(low_has + high_has) * wp.float64(0.5)
    return minus / (minus + plus)

