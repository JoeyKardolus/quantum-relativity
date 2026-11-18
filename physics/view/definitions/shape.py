"""Shape: where a place in a cell's block lies in the picture. A cell has no shape of its own.

The picture is the flat cut through the cells' middles, each cell as deep as its own size.
An energy volume is round: its block is drawn round, as wide as its size. A joint fills
what its two round volumes leave between them: from the middle of the one to the middle of
the other, out to where the next joints begin (a kite as wide as the volumes stand apart),
less the two round volumes. So it is thinnest on the line between the two middles and
widens toward the corners, where joints meet. A cell's `shape` holds its middle, then for
a joint (how far apart its two volumes' middles stand, the radius of the one before and of
the one after), then (0, its axis, minus its size); for a volume (its size, 0, 0), (0, 0, 1).
Picture: the grid's second direction runs right, its first down, its third toward the eye.
"""
import warp as wp


@wp.func
def drawn_place(inside: wp.vec3, shape: wp.mat33) -> wp.vec3:
    """A place in a block (each direction -0.5 to 0.5) as a place in the picture, in cell edges."""
    if shape[2][2] > 0.0:  # a volume: the square's rim goes to the round's
        widest = wp.max(wp.abs(inside[0]), wp.abs(inside[1]))
        scale = widest / wp.max(wp.sqrt(inside[0] * inside[0] + inside[1] * inside[1]), 1.0e-6)
        return shape[0] + shape[1][0] * wp.vec3(scale * inside[1], -scale * inside[0], inside[2])
    axis = int(shape[2][1])
    reach = wp.max(shape[1][0], 1.0e-6)  # the two middles' distance; the cell's middle is midway
    side, deep = inside[1 - axis] * reach, -inside[2] * shape[2][2]
    left = 1.0 - 2.0 * wp.abs(side) / reach  # 1 on the line between the middles, 0 at the corner
    low = wp.max(-0.5 * reach * left,
                 wp.sqrt(wp.max(shape[1][1] * shape[1][1] - side * side, 0.0)) - 0.5 * reach)
    high = wp.min(0.5 * reach * left,
                  0.5 * reach - wp.sqrt(wp.max(shape[1][2] * shape[1][2] - side * side, 0.0)))
    along = low + (inside[axis] + 0.5) * wp.max(high - low, 0.0)
    if axis == 1:
        return shape[0] + wp.vec3(along, -side, deep)
    return shape[0] + wp.vec3(side, -along, deep)
