"""Face step (Warp func): the six face neighbours of a Planck box, as steps.

Contract: `face_step(face)` for face 0..5 returns +x, -x, +y, -y, +z, -z as a vec3i;
face 2 * axis is the step up along `axis`.
"""
from __future__ import annotations

import warp as wp


@wp.func
def face_step(face: int) -> wp.vec3i:
    """The step to face neighbour `face` (0..5: +x, -x, +y, -y, +z, -z)."""
    step = wp.vec3i(0, 0, 0)
    axis = face // 2
    if face % 2 == 0:
        step[axis] = 1
    else:
        step[axis] = -1
    return step
