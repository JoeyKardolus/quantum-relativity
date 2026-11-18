"""Paint: the picture of one cell, its fluid as a faint red glow and its tracers on top.

Every dot of the picture looks along one line through the cell's block and gathers red
by the fluid's packing on the way (a box's quanta over the cell's mean a box, read
smoothly between the middles of the boxes). Thin fluid stays dark, so the swirl's open
core is a dark tube along its axis. `GLOW` only draws; it was set by eye once
(2026-11-18) and is not moved again.
"""
import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.store import peek
from physics.engine.axiom_3_structure.logic.boxes_of import boxes_of
from physics.view.definitions.camera import FOCAL, camera
from physics.view.logic.draw_tracers import draw_tracers

RED = wp.constant(wp.vec3(255.0, 38.0, 26.0))
GLOW = 0.3    # a line one cell edge long through fluid at the mean packing glows this much
SAMPLES = 64  # looks along one line


@wp.func
def quanta_at(spot: wp.vec3, box_quanta: wp.array2d(dtype=wp.int32), slot: int, across: int):
    """The quanta a box at a spot of the block (the cube from -0.5 to 0.5), smooth between boxes."""
    place = (spot + wp.vec3(0.5)) * float(across) - wp.vec3(0.5)
    found = float(0.0)
    for corner in range(8):
        weight = float(1.0)
        box = int(0)
        for axis in range(3):
            far = (corner >> axis) & 1
            share = place[axis] - wp.floor(place[axis])
            weight *= 1.0 - share + float(far) * (2.0 * share - 1.0)
            box = box * across + wp.clamp(int(wp.floor(place[axis])) + far, 0, across - 1)
        found += weight * float(box_quanta[slot, box])
    return found


@wp.kernel
def _glow(box_quanta: wp.array2d(dtype=wp.int32), slot: wp.int32, across: wp.int32,
          mean: wp.float32, eye: wp.vec3, right: wp.vec3, up: wp.vec3, forward: wp.vec3,
          picture: wp.array2d(dtype=wp.vec3)):
    row, column = wp.tid()
    half = 0.5 * float(picture.shape[0])
    line = wp.normalize(forward * FOCAL + right * ((float(column) + 0.5) / half
                        - float(picture.shape[1]) / float(picture.shape[0]))
                        + up * (1.0 - (float(row) + 0.5) / half))
    enters = float(0.0)  # how far along the line the cube begins and ends
    leaves = float(1.0e30)
    for axis in range(3):
        one_side = (-0.5 - eye[axis]) / line[axis]
        other_side = (0.5 - eye[axis]) / line[axis]
        enters = wp.max(enters, wp.min(one_side, other_side))
        leaves = wp.min(leaves, wp.max(one_side, other_side))
    stride = wp.max(leaves - enters, 0.0) / float(SAMPLES)
    light = float(0.0)
    for sample in range(SAMPLES):
        spot = eye + line * (enters + (float(sample) + 0.5) * stride)
        light += GLOW * quanta_at(spot, box_quanta, slot, across) / mean * stride
    picture[row, column] = RED * wp.min(light, 1.0)


def paint(store, cell: int, tracers, yaw: float, pitch: float, distance: float,
          width: int, height: int) -> np.ndarray:
    """The cell's picture as (height, width, 3) bytes, top row first; `distance` in cell edges."""
    box_quanta, _phase, slot = boxes_of(store, cell)
    view = [wp.vec3(*each) for each in camera(yaw, pitch, distance)]
    picture = wp.zeros((height, width), dtype=wp.vec3, device=box_quanta.device)
    mean = abs(peek(store.cell_quanta, cell)) / store.across ** 3
    wp.launch(_glow, dim=(height, width), device=box_quanta.device,
              inputs=[box_quanta, slot, store.across, mean, *view, picture])
    if tracers is not None:
        draw_tracers(tracers, store.across, view, picture)
    return np.minimum(picture.numpy(), 255.0).astype(np.uint8)
