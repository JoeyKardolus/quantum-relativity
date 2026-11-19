"""Line streaks: the tails of a scene's cells as the card draws them, each cell in its shape.

A tracer's second number says which slot's cell it is in (`across` a cell); `shape.py`
lays the cell's block out where and as wide as `placed` says.

One rule for both colours: tracers are laid by the quanta, so each stands for as much
energy, and every mark is drawn as solid as every other (`AMOUNT`, by its even share of
its cell's room). Colour so gathers where energy is packed, moving or still: a still
tracer is a dot, a moving one a streak. A joint's marks count below 0: blue.

A tracer that crossed a joint leaves its streak from the one volume to the other, through
the blue.
"""
import warp as wp

from physics.view.definitions.shape import drawn_place, even_share
from physics.view.logic.tracers import Tracers

AMOUNT = 0.05  # how solid one mark is drawn; only draws


@wp.kernel
def _spots(tail: wp.array2d(dtype=wp.vec3), head: wp.int32, across: wp.int32,
           placed: wp.array(dtype=wp.mat33), drawn: wp.array2d(dtype=wp.vec4)):
    age, tracer = wp.tid()  # 0 is the oldest mark; the last is where the tracer is now
    mark = tail[(head + 1 + age) % tail.shape[0], tracer]
    cell = wp.clamp(int(mark[1]) / across, 0, placed.shape[0] - 1)
    inside = wp.vec3(mark[0], mark[1] - float(cell * across), mark[2]) / float(across) - wp.vec3(0.5)
    spot = (drawn_place(inside, placed[cell]) + wp.vec3(0.5)) * float(across)
    counts = wp.sign(placed[cell][2][2]) * AMOUNT * even_share(inside, placed[cell])
    drawn[age, tracer] = wp.vec4(spot[0], spot[1], spot[2], counts)


def line_streaks(tracers: Tracers, across: int, placed: wp.array) -> wp.array:
    """(marks, tracers) of four floats: the place in the picture, how solid (below 0: blue)."""
    drawn = wp.empty(tracers.tail.shape, dtype=wp.vec4, device=tracers.tail.device)
    wp.launch(_spots, dim=drawn.shape, device=drawn.device,
              inputs=[tracers.tail, tracers.head, across, placed, drawn])
    return drawn
