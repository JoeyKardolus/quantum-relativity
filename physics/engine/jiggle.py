"""Jiggle: how hard the energy fluid jiggles inside each energy volume, per quantum, its heat.

Contract: `jiggle(state)` returns (spread, weight), two float64 arrays (universes,
capacity). Over every face between two plus boxes of the same cell (round the edges when
the universe wraps), with n and m their quanta and d the phase difference across the
face brought into (-pi, pi], weighted w = (n + m) / 2: per axis, spread sums
sum(w d**2) - sum(w d)**2 / sum(w), the weighted spread of d about the cell's mean
along that axis, over the three axes; weight sums w over all faces. With hbar = box
mass = 1 the fluid's speed along an axis is its phase difference per box, so spread /
weight is the quanta-weighted mean square of the speed's undirected part along one axis
(the cell's mean flow, as when the whole cell falls, taken out): the energy of the
fluid's own jiggling per quantum, its heat (bible: energy is an always-moving fluid;
friction turns directed flow into jiggling). Crossing heat raises it (cross.py
randomises arriving phases). Joints and empty boxes add nothing. Read only; the state is
not changed. Summed in whole units of 1 / UNIT with int64 atomics, so the sums do not
depend on the order the GPU adds them in (one seed, one run). cross.py pools the two
volumes of a joint.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.face_step import face_step
from physics.engine.reach import reach
from physics.engine.state import State

PI = 3.14159265358979
UNIT = 1048576.0  # numerics: fixed point for order-free sums, 2**20 per unit


@wp.kernel
def _faces(quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32), owner: wp.array3d(dtype=wp.int32),
           side: int, wrap: int, squares: wp.array3d(dtype=wp.int64), firsts: wp.array3d(dtype=wp.int64),
           weights: wp.array3d(dtype=wp.int64)):
    ti, tj, tk = wp.tid()
    held = quanta[ti, tj, tk]
    if held <= 0:
        return
    home = ti // side
    here = wp.vec3i(ti, tj, tk)
    cell = owner[ti, tj, tk]
    for axis in range(3):
        there, valid = reach(here, face_step(2 * axis), home, side, wrap)
        if valid and owner[there[0], there[1], there[2]] == cell:
            other = quanta[there[0], there[1], there[2]]
            if other > 0:
                turn = phase[there[0], there[1], there[2]] - phase[ti, tj, tk]
                turn = turn - 2.0 * PI * wp.floor((turn + PI) / (2.0 * PI))
                share = 0.5 * (float(held) + float(other))
                wp.atomic_add(squares, home, cell, axis, wp.int64(wp.round(share * turn * turn * UNIT)))
                wp.atomic_add(firsts, home, cell, axis, wp.int64(wp.round(share * turn * UNIT)))
                wp.atomic_add(weights, home, cell, axis, wp.int64(wp.round(share * UNIT)))


def jiggle(state: State) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's quanta-weighted spread of in-cell phase differences about its mean flow, and the weights."""
    shape = (state.universes, state.capacity, 3)
    device = state.quanta.device
    squares, firsts, weights = (wp.zeros(shape, dtype=wp.int64, device=device) for _ in range(3))
    wp.launch(_faces, dim=state.shape, device=device,
              inputs=[state.quanta, state.phase, state.owner, state.side, int(state.wrap), squares, firsts, weights])
    square, first, weight = (values.numpy() / UNIT for values in (squares, firsts, weights))
    spread = square - np.where(weight > 0.0, first ** 2 / np.maximum(weight, 1e-300), 0.0)
    return np.maximum(spread, 0.0).sum(axis=-1), weight.sum(axis=-1)
