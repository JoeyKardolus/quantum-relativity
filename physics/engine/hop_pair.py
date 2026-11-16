"""Hop pair (Warp func): one exact pair rotation between two face-neighbouring boxes of one cell, in whole quanta.

Contract: `hop_pair(quanta, phase, here, there, tau, seed, box_index)` turns the fluid of
the boxes `here` and `there` (both occupied, same cell: the caller checks) by one exact
pair rotation with turn -tau (pair_rotation.py), hands the whole quanta it moves from
`here` to `there` (signs kept, no box below 1 quantum) and writes both new phases, in
(-pi, pi]. The rounding's uniform comes from the hash of `seed` and `box_index` (the
place of `here` in its universe). Shared by hop.py (every pair of a pass at once) and
later_substeps.py (cell by cell), so both move exactly the same quanta.
"""
from __future__ import annotations

import warp as wp

from physics.engine.pair_rotation import pair_rotation

PI = 3.14159265358979


@wp.func
def _wrap(angle: float) -> float:
    """The angle brought into (-pi, pi]."""
    return angle - 2.0 * PI * wp.floor((angle + PI) / (2.0 * PI))


@wp.func
def _signed(sign_of: int, amount: int) -> int:
    """`amount` quanta with the sign of `sign_of`."""
    if sign_of < 0:
        return -amount
    return amount


@wp.func
def hop_pair(quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32), here: wp.vec3i,
             there: wp.vec3i, tau: float, seed: int, box_index: int):
    """Rotate the pair's fluid by -tau and hand over the whole quanta that moved."""
    near = quanta[here[0], here[1], here[2]]
    far = quanta[there[0], there[1], there[2]]
    rng = wp.rand_init(seed, box_index)
    moved, near_new, far_new = pair_rotation(wp.abs(near), wp.abs(far), phase[here[0], here[1], here[2]],
                                             phase[there[0], there[1], there[2]], -tau, wp.randf(rng))
    quanta[here[0], here[1], here[2]] = _signed(near, wp.abs(near) - moved)
    quanta[there[0], there[1], there[2]] = _signed(far, wp.abs(far) + moved)
    phase[here[0], here[1], here[2]] = _wrap(near_new)
    phase[there[0], there[1], there[2]] = _wrap(far_new)
