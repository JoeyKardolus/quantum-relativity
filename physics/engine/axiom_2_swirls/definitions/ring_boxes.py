"""Ring boxes: a cell's block of boxes laid as three ring whirls, one for each direction.

A ring whirl is fluid going round an empty core that is a ring closed inside the cell: up
through the cell's middle along the ring's direction, out over the side the direction
points to, back round the outside and in again from the other side. The boxes are not
space; they are numbered by where they lie in that roll. Ring `ring` (0, 1, 2: its
direction) holds a third of the block. A box lies `deep` layers out from the core (the
last layer is the cell's outside), `around` the core (the whirl's whole turn: `across`
boxes that close on themselves) and `along` the ring (closed too). Round the core the
first and the last box lie either side of the jet through the middle; the cap that faces
out along the direction follows the first, and the cap that faces back is its mirror.
"""
import warp as wp


@wp.func
def box_at(across: wp.int32, ring: wp.int32, deep: wp.int32, around: wp.int32,
           along: wp.int32) -> wp.int32:
    """The number of the box of ring `ring`, `deep` layers out, `around` the core, `along` the ring."""
    return ((ring * (across / 3) + deep) * across + around) * across + along


@wp.func
def cap_box(across: wp.int32, ring: wp.int32, row: wp.int32, after: wp.int32) -> wp.int32:
    """An outside box of ring `ring` that faces a neighbour along its direction: the one
    after (`after` 1) or before (0). A cap has `across * (across / 6)` boxes, numbered `row`."""
    around = across / 6 + row / across
    if after == 0:
        around = across - 1 - around  # the mirror: the cap that faces back
    return box_at(across, ring, across / 3 - 1, around, row % across)


def cap_boxes(across: int, ring: int, after: int):
    """On the host: the numbers of the boxes of one cap, in the order of their rows."""
    rows = range(across * (across // 6))
    around = [across // 6 + row // across for row in rows]
    around = around if after else [across - 1 - one for one in around]
    return [((ring * (across // 3) + across // 3 - 1) * across + one) * across + row % across
            for one, row in zip(around, rows)]
