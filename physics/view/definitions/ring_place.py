"""Ring place: where a box of a ring whirl lies in a ball 1 wide, and how much room it has.

A cell's block is three ring whirls, ring r about direction r (`ring_boxes.py`). In a
ring's own half-plane (h out from its axis, z along it) the core is the circle
h = R / sqrt(2), z = 0: the ring whirl that fills a ball (Hill's). A box `around` the core
lies at that share of a turn from the direction toward the axis, turning toward +z, so a
volume's fluid goes up the jet through the middle, out over the front cap and down the
outside. From the core outward the layers share the way to the ball's rim or the axis
along that ray, after the empty core, which takes `CORE` of the way: the innermost ring of
boxes is `across` equal boxes round (radius across / 2 pi boxes), then across / 3 layers.
`along` is the angle about the axis. Marks are laid evenly box for box, so `room` says
how much a mark counts for the colour to lie evenly in the ball.
"""
import math

import warp as wp

CORE = 3.0 / (3.0 + 2.0 * math.pi)  # the empty core's share of the way from core to rim
BALL = 0.5235987755982988           # the room of a ball 1 wide: pi over 6
AT = 0.5 / math.sqrt(2.0)           # the core circle's radius in a ball 1 wide


@wp.func
def ring_place(inside: wp.vec3) -> wp.vec4:
    """A place in a block (layer, around, along, each -0.5 to 0.5) as a place in a ball 1 wide,
    and the room a box has there over an even share."""
    ring = wp.clamp(int((inside[0] + 0.5) * 3.0), 0, 2)
    deep = (inside[0] + 0.5) * 3.0 - float(ring)
    turn, about = wp.tau * (inside[1] + 0.5), wp.tau * (inside[2] + 0.5)
    reach = AT * wp.cos(turn) + wp.sqrt(0.25 - AT * AT * wp.sin(turn) * wp.sin(turn))  # to the rim
    if wp.cos(turn) > 1.0e-6:
        reach = wp.min(reach, AT / wp.cos(turn))  # or to the axis
    out = (CORE + (1.0 - CORE) * deep) * reach
    from_axis = AT - out * wp.cos(turn)
    spot = wp.vec3(0.0)
    spot[ring] = out * wp.sin(turn)
    spot[(ring + 1) % 3] = from_axis * wp.cos(about)
    spot[(ring + 2) % 3] = from_axis * wp.sin(about)
    room = from_axis * out * (1.0 - CORE) * reach * wp.tau * wp.tau / BALL
    return wp.vec4(spot[0], spot[1], spot[2], room)
