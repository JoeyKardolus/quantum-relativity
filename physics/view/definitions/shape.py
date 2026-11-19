"""Shape: where a place in a cell's block lies in the picture. A cell has no shape of its own.

An energy volume is round: its three ring whirls are drawn in a ball as wide as its size
(`ring_place.py`). The joints fill all the room the round volumes leave, each the room
nearest its own line between two middles: two pyramids, their points in the two volumes'
middles and their shared base midway, less the two balls. A joint's ball place is spread
to a cube and each half of the cube fans out from the volume it faces, from that ball's
rim to the base, so its ring for its own direction lies along its line. A cell's `shape`
holds its middle, then for a joint (how far apart its volumes' middles stand along its
axis, the radius of the one before, of the one after) and (its lean toward each of the two
other directions, -1 - its axis); for a volume (its size, 0, 0), (0, 0, 1); a minus cell
with no volumes to lie between is a ball too: (0, 0, -0.5).
Picture: the cells' second direction runs right, its first down, its third toward the eye.
"""
import warp as wp

from physics.view.definitions.ring_place import BALL, ring_place


@wp.func
def _pressed(spot: wp.vec3) -> float:
    """How much a ball's place is pressed together from the cube's: 1 on the axes."""
    widest = wp.max(wp.abs(spot[0]), wp.max(wp.abs(spot[1]), wp.abs(spot[2])))
    return widest / wp.max(wp.length(spot), 1.0e-6)


@wp.func
def drawn_place(inside: wp.vec3, shape: wp.mat33) -> wp.vec3:
    """A place in a block (each number -0.5 to 0.5) as a place in the picture, in cell edges."""
    ball = ring_place(inside)
    spot = wp.vec3(ball[0], ball[1], ball[2])
    if shape[2][2] > -0.75:  # a ball as wide as its size
        spot = spot * shape[1][0]
    else:
        cube = spot / _pressed(spot)  # the ball's rim goes to the cube's
        axis = wp.max(int(-shape[2][2]) - 1, 0)
        along = wp.vec3(float(axis == 0), float(axis == 1), float(axis == 2))
        side, radius = float(-1.0), shape[1][1]  # the half that faces the volume before
        if cube[axis] >= 0.0:
            side, radius = 1.0, shape[1][2]
        middle = along * (side * 0.5 * shape[1][0])  # of the volume this half faces
        ray = (cube - along * cube[axis]) * shape[1][0] - middle  # from it to the base
        reach = wp.length(ray)
        part = 1.0 - 2.0 * wp.abs(cube[axis])  # 0 at the ball's rim, 1 at the base
        rim = wp.min(radius, reach)
        out = wp.cbrt(rim * rim * rim * (1.0 - part) + reach * reach * reach * part)
        spot = middle + ray * (out / reach)
        lean = wp.vec3(shape[2][0], shape[2][1], 0.0)
        if axis == 0:
            lean = wp.vec3(0.0, shape[2][0], shape[2][1])
        elif axis == 1:
            lean = wp.vec3(shape[2][0], 0.0, shape[2][1])
        spot += lean * spot[axis]
    return shape[0] + wp.vec3(spot[1], -spot[0], spot[2])


@wp.func
def even_share(inside: wp.vec3, shape: wp.mat33) -> float:
    """How much a mark counts so that the colour lies evenly in a cell's room: the room its
    box has in the ball, and for a joint the ball spread out to the cube again."""
    ball = ring_place(inside)
    if shape[2][2] > -0.75:
        return ball[3]
    return ball[3] * BALL / wp.pow(_pressed(wp.vec3(ball[0], ball[1], ball[2])), 3.0)
