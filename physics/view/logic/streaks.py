"""Streaks: the tracers' tails as the numbers the card draws, worked out on the card.

For every mark of every tail: its place, and how solid it is drawn: by the energy moving
there, the square of how far the tracer went to reach it. Oldest mark first, so a tail's
marks follow each other and no line joins its two ends. `FULL` was set by eye once.
"""
import warp as wp

from physics.view.logic.tracers import Tracers

FULL = 0.6  # fluid that goes this many boxes in one mark is drawn fully solid


@wp.kernel
def _streak(tail: wp.array2d(dtype=wp.vec3), head: wp.int32,
            drawn: wp.array2d(dtype=wp.vec4)):
    age, tracer = wp.tid()  # 0 is the oldest mark; the last is where the tracer is now
    marks = tail.shape[0]
    place = tail[(head + 1 + age) % marks, tracer]
    gone = float(0.0)
    if age > 0:
        gone = wp.length(place - tail[(head + age) % marks, tracer])
    drawn[age, tracer] = wp.vec4(place[0], place[1], place[2],
                                 wp.min(gone * gone / (FULL * FULL), 1.0))


def streaks(tracers: Tracers) -> wp.array:
    """(marks, tracers) of four floats on the tracers' device: the place in boxes, how solid."""
    drawn = wp.empty(tracers.tail.shape, dtype=wp.vec4, device=tracers.tail.device)
    wp.launch(_streak, dim=drawn.shape, device=drawn.device,
              inputs=[tracers.tail, tracers.head, drawn])
    return drawn
