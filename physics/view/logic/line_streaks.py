"""Line streaks: the tails of a scene's cells as the card draws them, each cell in its shape.

A tracer's second number says which slot's cell it is in (`across` a cell); `shape.py`
lays the cell's block out where and as wide as `placed` says.

One rule for both colours: tracers are laid by the quanta, so each stands for as much
energy, and every mark is drawn as solid as every other (`AMOUNT`, by its even share of
its cell's room). Colour so gathers where energy is packed, moving or still: a still
tracer is a dot, a moving one a streak. A joint's marks count below 0: blue.

Every mark is drawn. In three dimensions (`web`) a joint's are drawn only where it is cut
(`in_the_web`): the blue is then a web between the balls and not a fog before them.
"""
import warp as wp

from physics.view.definitions.shape import drawn_place, even_share
from physics.view.logic.tracers import Tracers

AMOUNT = 0.05  # how solid one mark is drawn; only draws


@wp.func
def in_the_web(inside: wp.vec3, shape: wp.mat33, across: wp.int32) -> float:
    """1 for a joint's mark in the two middle layers of boxes across either direction across
    its length (the joint cut both ways along its length), and for every mark of a volume."""
    axis = int(shape[2][1])
    near = wp.min(wp.abs(inside[(axis + 1) % 3]), wp.abs(inside[(axis + 2) % 3]))
    return float(shape[2][2] > 0.0 or near * float(across) < 1.0)


@wp.kernel
def _spots(tail: wp.array2d(dtype=wp.vec3), head: wp.int32, across: wp.int32, web: wp.int32,
           placed: wp.array(dtype=wp.mat33), drawn: wp.array2d(dtype=wp.vec4)):
    age, tracer = wp.tid()  # 0 is the oldest mark; the last is where the tracer is now
    mark = tail[(head + 1 + age) % tail.shape[0], tracer]
    cell = wp.clamp(int(mark[1]) / across, 0, placed.shape[0] - 1)
    inside = wp.vec3(mark[0], mark[1] - float(cell * across), mark[2]) / float(across) - wp.vec3(0.5)
    spot = (drawn_place(inside, placed[cell]) + wp.vec3(0.5)) * float(across)
    counts = wp.sign(placed[cell][2][2]) * AMOUNT * even_share(inside, placed[cell])
    if web > 0:
        counts *= in_the_web(inside, placed[cell], across)
    drawn[age, tracer] = wp.vec4(spot[0], spot[1], spot[2], counts)


@wp.kernel
def _not_round_the_wrap(across: wp.int32, drawn: wp.array2d(dtype=wp.vec4)):
    age, tracer = wp.tid()
    spot = drawn[age, tracer]
    # a mark a cell's edge or more from the next is the same place the other way round the
    # grid, where it wraps: that way is not drawn
    after = drawn[wp.min(age + 1, drawn.shape[0] - 1), tracer] - spot
    before = drawn[wp.max(age - 1, 0), tracer] - spot
    if wp.max(wp.length(wp.vec3(after[0], after[1], after[2])),
              wp.length(wp.vec3(before[0], before[1], before[2]))) > float(across):
        drawn[age, tracer] = wp.vec4(spot[0], spot[1], spot[2], 0.0)


def line_streaks(tracers: Tracers, across: int, placed: wp.array, web: bool = False) -> wp.array:
    """(marks, tracers) of four floats: the place in the picture, how solid (below 0: blue)."""
    drawn = wp.empty(tracers.tail.shape, dtype=wp.vec4, device=tracers.tail.device)
    wp.launch(_spots, dim=drawn.shape, device=drawn.device,
              inputs=[tracers.tail, tracers.head, across, int(web), placed, drawn])
    wp.launch(_not_round_the_wrap, dim=drawn.shape, device=drawn.device, inputs=[across, drawn])
    return drawn
