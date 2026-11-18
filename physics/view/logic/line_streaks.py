"""Line streaks: the tails of a line of cells as the card draws them, each cell at its own size.

A cell has no shape, only a size by the size law and a place in the line, so the cells
are drawn touching, one after the other from left to right, each a cube as long as its
size. The line runs along the boxes' second direction; it is turned to lie across the
picture. A joint's marks carry their way gone below 0, which the drawing makes blue.

One rule for both colours: a streak is as solid as its fluid is fast beside its own cell's
fluid. The way gone is counted in the cell's own boxes (two facing boxes of two volumes
lie next to each other), and fluid at its cell's mean speed is drawn with `AT_MEAN`.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.size_law import size_of
from physics.view.logic.tracers import Tracers

AT_MEAN = 0.12  # the way gone that fluid at its cell's mean speed is drawn with; only draws


def placed_line(store, cells: int) -> wp.array:
    """For each cell of the line: its middle along it, its size, and 1 (plus) or -1 (minus)."""
    quanta = store.cell_quanta.numpy()[:cells]
    size = np.array([size_of(int(held), store.across) for held in quanta])
    middle = np.cumsum(size) - size / 2 - size.sum() / 2
    return wp.array(np.stack([middle, size, np.where(quanta < 0, -1.0, 1.0)], axis=1),
                    dtype=wp.vec3, device=store.cell_quanta.device)


@wp.func
def _drawn_place(place: wp.vec3, across: wp.int32, placed: wp.array(dtype=wp.vec3)) -> wp.vec3:
    """A place in the line's boxes as a place in the picture, in boxes of a cell 1 long."""
    cell = wp.clamp(int(place[1]) / across, 0, placed.shape[0] - 1)
    half, size = 0.5 * float(across), placed[cell][1]
    along = placed[cell][0] * float(across) + (place[1] - float(cell * across) - half) * size
    return wp.vec3(along + half, (half - place[0]) * size + half, (place[2] - half) * size + half)


@wp.kernel
def _way_gone(tail: wp.array2d(dtype=wp.vec3), head: wp.int32, across: wp.int32,
              placed: wp.array(dtype=wp.vec3), drawn: wp.array2d(dtype=wp.vec4),
              speed: wp.array(dtype=wp.vec2)):
    age, tracer = wp.tid()  # 0 is the oldest mark; the last is where the tracer is now
    marks = tail.shape[0]
    mark = tail[(head + 1 + age) % marks, tracer]
    spot = _drawn_place(mark, across, placed)
    cell = wp.clamp(int(mark[1]) / across, 0, placed.shape[0] - 1)
    gone = float(0.0)
    if age > 0:
        way = mark - tail[(head + age) % marks, tracer]
        came_from = wp.clamp(int(mark[1] - way[1]) / across, 0, placed.shape[0] - 1)
        way[1] -= float(across * ((cell - came_from) / 2))  # over a joint: the next box
        gone = wp.length(way)
        wp.atomic_add(speed, cell, wp.vec2(gone * gone, 1.0))
    drawn[age, tracer] = wp.vec4(spot[0], spot[1], spot[2], gone)


@wp.kernel
def _beside_mean(tail: wp.array2d(dtype=wp.vec3), head: wp.int32, across: wp.int32,
                 placed: wp.array(dtype=wp.vec3), speed: wp.array(dtype=wp.vec2),
                 drawn: wp.array2d(dtype=wp.vec4)):
    age, tracer = wp.tid()
    mark = tail[(head + 1 + age) % tail.shape[0], tracer]
    cell = wp.clamp(int(mark[1]) / across, 0, placed.shape[0] - 1)
    mean = wp.sqrt(speed[cell][0] / wp.max(speed[cell][1], 1.0))
    spot = drawn[age, tracer]
    drawn[age, tracer] = wp.vec4(spot[0], spot[1], spot[2],
                                 placed[cell][2] * AT_MEAN * spot[3] / wp.max(mean, 1.0e-6))


def line_streaks(tracers: Tracers, across: int, placed: wp.array) -> wp.array:
    """(marks, tracers) of four floats: the place in the picture, how solid (below 0: minus)."""
    drawn = wp.empty(tracers.tail.shape, dtype=wp.vec4, device=tracers.tail.device)
    speed = wp.zeros(placed.shape[0], dtype=wp.vec2, device=drawn.device)
    wp.launch(_way_gone, dim=drawn.shape, device=drawn.device,
              inputs=[tracers.tail, tracers.head, across, placed, drawn, speed])
    wp.launch(_beside_mean, dim=drawn.shape, device=drawn.device,
              inputs=[tracers.tail, tracers.head, across, placed, speed, drawn])
    return drawn
