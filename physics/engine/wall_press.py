"""Wall press: how hard the energy volumes' fluid presses on each joint, read where the fluid meets the joint's boxes.

Contract: `wall_press(state, settings)` reads state.cell_quanta and state.cell_boxes
(cell_count.py) and returns, per universe and cell id (float64 numpy), the mean over
every face where a plus box meets a joint box of that plus box's fluid pressure,
push * (|quanta| / packing - 1), packing its volume's quanta per box (fluid_potential.py:
the fluid's own pressure, measured from its volume's mean); 0 for a cell with no such
face. Where a volume's fluid is denser than its mean it presses the joint smaller, where
thinner it draws it out (owner, 2026-11-16: energy is always moving, a whirl holds its
core open by the fluid's own pressure; a fresh joint gains as the volumes press on it).
joint_swing.py takes it as a load against the joint's push, like the pull. A volume as a
whole pushes nothing: its fluid's pressure averages to 0 over its boxes. The sum is
taken in whole millionths (int64), so it is the same in any order.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.face_step import face_step
from physics.engine.reach import reach
from physics.engine.settings import Settings
from physics.engine.state import State

PRESS_UNIT = 1.0e6


@wp.kernel
def _press(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32),
           packings: wp.array2d(dtype=wp.float32), push: float, side: int, wrap: int,
           sums: wp.array2d(dtype=wp.int64), faces: wp.array2d(dtype=wp.int64)):
    ti, tj, tk = wp.tid()
    if quanta[ti, tj, tk] >= 0:
        return
    joint = owner[ti, tj, tk]
    if joint < 0:
        return
    home = ti // side
    here = wp.vec3i(ti, tj, tk)
    for face in range(6):
        there, valid = reach(here, face_step(face), home, side, wrap)
        if valid:
            held = quanta[there[0], there[1], there[2]]
            if held > 0:
                packing = packings[home, owner[there[0], there[1], there[2]]]
                if packing > 0.0:
                    pressure = push * (float(held) / packing - 1.0)
                    wp.atomic_add(sums, home, joint, wp.int64(wp.round(pressure * PRESS_UNIT)))
                    wp.atomic_add(faces, home, joint, wp.int64(1))


def wall_press(state: State, settings: Settings) -> np.ndarray:
    """Each joint's mean fluid pressure from the plus boxes on its faces."""
    shape = (state.universes, state.capacity)
    device = state.quanta.device
    packing = (state.cell_quanta / np.maximum(state.cell_boxes, 1)).astype(np.float32)
    sums = wp.zeros(shape, dtype=wp.int64, device=device)
    faces = wp.zeros(shape, dtype=wp.int64, device=device)
    wp.launch(_press, dim=state.shape, device=device,
              inputs=[state.quanta, state.owner, wp.array(packing, dtype=wp.float32, device=device),
                      float(settings.push), state.side, int(state.wrap), sums, faces])
    return sums.numpy() / PRESS_UNIT / np.maximum(faces.numpy(), 1)
