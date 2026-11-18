"""Line streaks: the tails of a line of cells as the card draws them, each cell at its own size.

A cell has no shape, only a size by the size law and a place in the line, so the cells
are drawn touching, one after the other from left to right, each a cube as long as its
size. The line runs along the boxes' second direction; it is turned to lie across the
picture. A joint's marks carry their way gone below 0, which the drawing makes blue.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.size_law import size_of
from physics.view.logic.tracers import Tracers


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
def _line_streak(tail: wp.array2d(dtype=wp.vec3), head: wp.int32, across: wp.int32,
                 placed: wp.array(dtype=wp.vec3), drawn: wp.array2d(dtype=wp.vec4)):
    age, tracer = wp.tid()  # 0 is the oldest mark; the last is where the tracer is now
    marks = tail.shape[0]
    mark = tail[(head + 1 + age) % marks, tracer]
    spot = _drawn_place(mark, across, placed)
    gone = float(0.0)
    if age > 0:
        gone = wp.length(spot - _drawn_place(tail[(head + age) % marks, tracer], across, placed))
    kind = placed[wp.clamp(int(mark[1]) / across, 0, placed.shape[0] - 1)][2]
    drawn[age, tracer] = wp.vec4(spot[0], spot[1], spot[2], kind * gone)


def line_streaks(tracers: Tracers, across: int, placed: wp.array) -> wp.array:
    """(marks, tracers) of four floats: the place in the picture, the way gone (below 0: minus)."""
    drawn = wp.empty(tracers.tail.shape, dtype=wp.vec4, device=tracers.tail.device)
    wp.launch(_line_streak, dim=drawn.shape, device=drawn.device,
              inputs=[tracers.tail, tracers.head, across, placed, drawn])
    return drawn
