"""Paint: the picture of one cell's fluid, as a red volume you can see into.

Every dot of the picture looks along one line through the cell's block. Each box on
it stops light by its packing (its quanta over the cell's mean a box): thin fluid is
see-through, packed fluid solid red. A box glows by its phase, `(1 + cos) / 2`, so the
swirl shows as a bright side going round: that is the phase's round, 92 steps at 6
across; the fluid itself goes round faster, in 71. `THICKNESS` and `DIM` only draw.
They were set by eye once (2026-11-18) and are not moved again.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.store import peek
from physics.engine.axiom_3_structure.logic.boxes_of import boxes_of
from physics.view.definitions.camera import FOCAL, camera

RED = wp.constant(wp.vec3(255.0, 38.0, 26.0))
THICKNESS = 2.0  # a line through the whole cell at the mean packing lets e^-2 of the light by
DIM = 0.25       # the glow of a box whose phase points away, so the whole volume stays in sight
SAMPLES = 96     # looks along one line


@wp.func
def line_through(row: int, column: int, rows: int, columns: int,
                 right: wp.vec3, up: wp.vec3, forward: wp.vec3) -> wp.vec3:
    """The direction the eye looks in through one dot of the picture."""
    half = 0.5 * float(rows)
    sideways = (float(column) + 0.5 - 0.5 * float(columns)) / half
    upward = (half - float(row) - 0.5) / half
    return wp.normalize(forward * FOCAL + right * sideways + up * upward)


@wp.func
def box_at(spot: wp.vec3, across: int) -> int:
    """The box at a spot in the block, which is drawn as the cube from -0.5 to 0.5."""
    place = (spot + wp.vec3(0.5)) * float(across)
    first = wp.clamp(int(place[0]), 0, across - 1)
    second = wp.clamp(int(place[1]), 0, across - 1)
    return (first * across + second) * across + wp.clamp(int(place[2]), 0, across - 1)


@wp.kernel
def _paint(box_quanta: wp.array2d(dtype=wp.int32), box_phase: wp.array2d(dtype=wp.float32),
           slot: wp.int32, across: wp.int32, mean: wp.float32,
           eye: wp.vec3, right: wp.vec3, up: wp.vec3, forward: wp.vec3,
           picture: wp.array3d(dtype=wp.uint8)):
    row, column = wp.tid()
    line = line_through(row, column, picture.shape[0], picture.shape[1], right, up, forward)
    enters = float(0.0)  # how far along the line the cube begins and ends
    leaves = float(1.0e30)
    for axis in range(3):
        one_side = (-0.5 - eye[axis]) / line[axis]
        other_side = (0.5 - eye[axis]) / line[axis]
        enters = wp.max(enters, wp.min(one_side, other_side))
        leaves = wp.min(leaves, wp.max(one_side, other_side))
    light = float(0.0)
    passes = float(1.0)  # the share of light that still gets through to the eye
    stride = (leaves - enters) / float(SAMPLES)
    for sample in range(SAMPLES):
        if leaves > enters:
            box = box_at(eye + line * (enters + (float(sample) + 0.5) * stride), across)
            stopped = 1.0 - wp.exp(-THICKNESS * float(box_quanta[slot, box]) / mean * stride)
            glow = 0.5 * (1.0 + wp.cos(box_phase[slot, box]))
            light += passes * stopped * (DIM + (1.0 - DIM) * glow)
            passes *= 1.0 - stopped
    for colour in range(3):
        picture[row, column, colour] = wp.uint8(RED[colour] * light)


def paint(store, cell: int, yaw: float, pitch: float, distance: float, width: int, height: int):
    """The cell's picture as (height, width, 3) bytes, top row first; `distance` in cell edges."""
    box_quanta, box_phase, slot = boxes_of(store, cell)
    eye, right, up, forward = (wp.vec3(*each) for each in camera(yaw, pitch, distance))
    picture = wp.zeros((height, width, 3), dtype=wp.uint8, device=box_quanta.device)
    mean = abs(peek(store.cell_quanta, cell)) / store.across ** 3
    wp.launch(_paint, dim=(height, width), device=box_quanta.device,
              inputs=[box_quanta, box_phase, slot, store.across, mean,
                      eye, right, up, forward, picture])
    return picture.numpy()
