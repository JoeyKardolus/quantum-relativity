"""Draw tracers: every tracer as a light dot with its tail fading behind it.

The tail is the way the tracer came, so in one still picture it shows which way and how
fast the fluid there goes. `LIGHT`, `HEAD`, `TAIL` and `PIECES` only draw; they were set
by eye once (2026-11-18).
"""
import warp as wp

from physics.view.definitions.camera import FOCAL
from physics.view.logic.tracers import Tracers

LIGHT = wp.constant(wp.vec3(255.0, 225.0, 150.0))
HEAD = 3     # the dot at the tracer itself is this many picture dots wide and high
TAIL = 0.6   # how bright the tail starts, of the head's light
PIECES = 12  # dots drawn along the way from one mark to the next


@wp.kernel
def _draw(tail: wp.array2d(dtype=wp.vec3), head: wp.int32, across: wp.float32,
          eye: wp.vec3, right: wp.vec3, up: wp.vec3, forward: wp.vec3,
          picture: wp.array2d(dtype=wp.vec3)):
    tracer, age, piece = wp.tid()
    marks = tail.shape[0]
    newer = tail[(head - age + marks) % marks, tracer]
    older = tail[(head - age - 1 + 2 * marks) % marks, tracer]
    along = float(piece) / float(PIECES)
    spot = (newer + (older - newer) * along) / across - wp.vec3(0.5) - eye
    depth = wp.dot(spot, forward)
    half = 0.5 * float(picture.shape[0])
    column = int(0.5 * float(picture.shape[1]) + FOCAL * wp.dot(spot, right) / depth * half)
    row = int(half - FOCAL * wp.dot(spot, up) / depth * half)
    fade = 1.0 - (float(age) + along) / float(marks - 1)
    wide = 1
    shine = TAIL * fade * fade
    if age == 0 and piece == 0:
        wide = HEAD
        shine = 1.0
    for down in range(wide):
        for over in range(wide):
            if (depth > 0.0 and row + down >= 0 and row + down < picture.shape[0]
                    and column + over >= 0 and column + over < picture.shape[1]):
                wp.atomic_add(picture, row + down, column + over, LIGHT * shine)


def draw_tracers(tracers: Tracers, across: int, view: list, picture: wp.array) -> None:
    """Add the tracers and their tails to a picture; `view` is the camera's four directions."""
    marks, count = tracers.tail.shape
    wp.launch(_draw, dim=(count, marks - 1, PIECES), device=picture.device,
              inputs=[tracers.tail, tracers.head, float(across), *view, picture])
