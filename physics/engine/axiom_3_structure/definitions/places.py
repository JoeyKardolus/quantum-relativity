"""Places: where a cell lies in the pattern the leak has made, as three whole numbers.

A cell has no geometry, only relations. To know which cells are neighbours, a cell's
number is its place: how many steps along each of the three directions (`side` numbers a
direction). Energy volumes lie at the even steps and the joint between two of them at the
odd step between, so a joint's direction and its two volumes follow by arithmetic and
nothing else is stored. This is bookkeeping only: a place holds nothing until a leak
makes a cell there, and it says nothing of sizes or distances. A store with `side` 0
keeps no places. `stride_of` serves a cell's block of boxes the same way.
"""
import numpy as np
import warp as wp


@wp.func
def stride_of(side: wp.int32, axis: wp.int32) -> wp.int32:
    """How far apart in number two neighbours along `axis` lie, `side` numbers a direction."""
    if axis == 0:
        return side * side
    if axis == 1:
        return side
    return 1


@wp.func
def beside(cell: wp.int32, side: wp.int32, axis: wp.int32, steps: wp.int32) -> wp.int32:
    """The cell `steps` steps along `axis` from `cell`; -1 past the room's end."""
    apart = stride_of(side, axis)
    at = cell / apart % side + steps
    if at < 0 or at >= side:
        return -1
    return cell + steps * apart


@wp.func
def joint_axis(cell: wp.int32, side: wp.int32) -> wp.int32:
    """The direction a joint's place lies along, its one odd step; -1 at a volume's place."""
    if side == 0:
        return -1
    for axis in range(3):
        if cell / stride_of(side, axis) % side % 2 == 1:
            return axis
    return -1


@wp.func
def joint_ends(joint: wp.int32, side: wp.int32, axis: wp.int32) -> wp.vec2i:
    """A joint's two volumes: the one a step before it along its axis, and the one after."""
    return wp.vec2i(joint - stride_of(side, axis), joint + stride_of(side, axis))


def steps_of(cells, side: int) -> np.ndarray:
    """On the host: the cells' steps along the three directions, (cells, 3)."""
    return np.stack(np.unravel_index(cells, (side,) * 3), axis=-1)
