"""Direction (Warp function): the six face directions of a Planck box, numbered 0 to 5.

Contract: `direction(face)` returns the unit step of face 0 to 5 as a vec3i: face 2a
is +1 along axis a, face 2a + 1 is -1 along it, so face ^ 1 is the opposite face.
"""
from __future__ import annotations

import warp as wp


@wp.func
def direction(face: int) -> wp.vec3i:
    """The unit step towards face 0..5: +x, -x, +y, -y, +z, -z."""
    step = wp.vec3i(0, 0, 0)
    step[face // 2] = 1 - 2 * (face % 2)
    return step
