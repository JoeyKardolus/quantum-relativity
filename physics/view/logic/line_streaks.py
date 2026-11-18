"""Line streaks: the tails of a scene's cells as the card draws them, each cell in its shape.

A tracer's second number says which slot's cell it is in (`across` a cell); `shape.py`
lays the cell's block out where and as wide as `placed` says. A joint's marks carry their
way gone below 0, which the drawing makes blue.

One rule for both colours: a streak is as solid as its fluid is fast beside its own cell's
fluid. The way gone is counted in the cell's own boxes (two facing boxes of two volumes
lie next to each other), and fluid at its cell's mean speed is drawn with `AT_MEAN`.
"""
import warp as wp

from physics.view.definitions.shape import drawn_place
from physics.view.logic.tracers import Tracers

AT_MEAN = 0.12  # the way gone that fluid at its cell's mean speed is drawn with; only draws


@wp.func
def _inside(mark: wp.vec3, across: wp.int32, cells: wp.int32) -> wp.vec3:
    """A mark's place inside its cell's block, each direction from -0.5 to 0.5."""
    cell = wp.clamp(int(mark[1]) / across, 0, cells - 1)
    return wp.vec3(mark[0], mark[1] - float(cell * across), mark[2]) / float(across) - wp.vec3(0.5)


@wp.kernel
def _way_gone(tail: wp.array2d(dtype=wp.vec3), head: wp.int32, across: wp.int32,
              placed: wp.array(dtype=wp.mat33), drawn: wp.array2d(dtype=wp.vec4),
              speed: wp.array(dtype=wp.vec2)):
    age, tracer = wp.tid()  # 0 is the oldest mark; the last is where the tracer is now
    marks = tail.shape[0]
    mark = tail[(head + 1 + age) % marks, tracer]
    cell = wp.clamp(int(mark[1]) / across, 0, placed.shape[0] - 1)
    inside = _inside(mark, across, placed.shape[0])
    spot = (drawn_place(inside, placed[cell]) + wp.vec3(0.5)) * float(across)
    gone = float(0.0)
    if age > 0:
        way = inside - _inside(tail[(head + age) % marks, tracer], across, placed.shape[0])
        # over a joint a tracer lands in the next volume's first box: the next box
        way -= wp.vec3(wp.round(way[0]), wp.round(way[1]), wp.round(way[2]))
        gone = wp.length(way) * float(across)
        wp.atomic_add(speed, cell, wp.vec2(gone * gone, 1.0))
    drawn[age, tracer] = wp.vec4(spot[0], spot[1], spot[2], gone)


@wp.kernel
def _beside_mean(tail: wp.array2d(dtype=wp.vec3), head: wp.int32, across: wp.int32,
                 placed: wp.array(dtype=wp.mat33), speed: wp.array(dtype=wp.vec2),
                 drawn: wp.array2d(dtype=wp.vec4)):
    age, tracer = wp.tid()
    mark = tail[(head + 1 + age) % tail.shape[0], tracer]
    cell = wp.clamp(int(mark[1]) / across, 0, placed.shape[0] - 1)
    mean = wp.sqrt(speed[cell][0] / wp.max(speed[cell][1], 1.0))
    spot = drawn[age, tracer]
    solid = wp.sign(placed[cell][2][2]) * AT_MEAN * spot[3] / wp.max(mean, 1.0e-6)
    # a mark a cell's edge or more from the next is the same place the other way round the
    # grid, where it wraps: that way is not drawn
    after = drawn[wp.min(age + 1, tail.shape[0] - 1), tracer] - spot
    before = drawn[wp.max(age - 1, 0), tracer] - spot
    if wp.max(wp.length(wp.vec3(after[0], after[1], after[2])),
              wp.length(wp.vec3(before[0], before[1], before[2]))) > float(across):
        solid = 0.0
    drawn[age, tracer] = wp.vec4(spot[0], spot[1], spot[2], solid)


def line_streaks(tracers: Tracers, across: int, placed: wp.array) -> wp.array:
    """(marks, tracers) of four floats: the place in the picture, how solid (below 0: minus)."""
    drawn = wp.empty(tracers.tail.shape, dtype=wp.vec4, device=tracers.tail.device)
    speed = wp.zeros(placed.shape[0], dtype=wp.vec2, device=drawn.device)
    wp.launch(_way_gone, dim=drawn.shape, device=drawn.device,
              inputs=[tracers.tail, tracers.head, across, placed, drawn, speed])
    wp.launch(_beside_mean, dim=drawn.shape, device=drawn.device,
              inputs=[tracers.tail, tracers.head, across, placed, speed, drawn])
    return drawn
