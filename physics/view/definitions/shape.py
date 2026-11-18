"""Shape: where a place in a cell's block lies in the picture. A cell has no shape of its own.

An energy volume is round: its block is drawn as a ball as wide as its size. The joints
fill all the room the round volumes leave, each the room nearest its own line between two
middles: two pyramids, their points in the two volumes' middles and their shared base
midway, less the two balls. So a joint is thinnest on the line between its volumes and
fat where four or eight volumes meet, and the joints that meet there fill it between them.
Each half of a joint's block fans out from the volume it faces, from that ball's rim to
the base, equal parts of the block along a ray getting equal room. A cell's `shape` holds
its middle, then for a joint (how far apart its volumes' middles stand, the radius of the
one before, of the one after) and (how far its second half is moved where the grid wraps,
its axis, -1); for a volume (its size, 0, 0), (0, 0, 1).
Picture: the grid's second direction runs right, its first down, its third toward the eye.
"""
import warp as wp


@wp.func
def drawn_place(inside: wp.vec3, shape: wp.mat33) -> wp.vec3:
    """A place in a block (each direction -0.5 to 0.5) as a place in the picture, in cell edges."""
    spot = inside
    if shape[2][2] > 0.0:  # a volume: the cube's rim goes to the ball's
        widest = wp.max(wp.abs(inside[0]), wp.max(wp.abs(inside[1]), wp.abs(inside[2])))
        spot = inside * (shape[1][0] * widest / wp.max(wp.length(inside), 1.0e-6))
    else:
        axis = int(shape[2][1])
        along = wp.vec3(float(axis == 0), float(axis == 1), float(axis == 2))
        side, radius = float(-1.0), shape[1][1]  # the half that faces the volume before
        if inside[axis] >= 0.0:
            side, radius = 1.0, shape[1][2]
        middle = along * (side * 0.5 * shape[1][0])  # of the volume this half faces
        ray = (inside - along * inside[axis]) * shape[1][0] - middle  # from it to the base
        reach = wp.length(ray)
        part = 1.0 - 2.0 * wp.abs(inside[axis])  # 0 at the ball's rim, 1 at the base
        rim = wp.min(radius, reach)
        out = wp.cbrt(rim * rim * rim * (1.0 - part) + reach * reach * reach * part)
        spot = middle + ray * (out / reach) + along * (wp.max(side, 0.0) * shape[2][0])
    return shape[0] + wp.vec3(spot[1], -spot[0], spot[2])
