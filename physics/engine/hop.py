"""Hop: the quantum fluid moves whole quanta between face-neighbouring Planck boxes of one cell, one exact pair rotation each.

Contract: `hop(state, settings, axis, parity, duration, pass_index, sub_index=0)` advances
the fluid's kinetic step (Schrodinger, hbar = box mass = 1: i dpsi/dt = -1/2 sum over
faces (psi_there - psi) + on-site terms, on_site.py) across every face along `axis`
whose lower box has that coordinate's `parity` in its own universe (round the edge too
when the universe wraps, state.wrap; the side is then even, so the pairs of one pass
still share no box). The pair runs only when both boxes belong to the same cell: inside
an energy volume and inside a joint the flow is frictionless (bible rule 4); between
cells energy moves only by crossing a joint (cross.py). The scheme is De Raedt's product
formula (1987): one exact pair rotation (pair_rotation.py) with turn -tau, tau =
duration * rate (the cell's local clock, state.rate), handing over whole quanta; the
rounding's uniform comes from a hash of the box's place in its universe, the universe's
key (state.py), the step, the substep and the pass, so a universe steps the same alone
or in a batch. With fluid substeps (substeps.py) universe u's pairs turn for duration /
state.substeps[u], only while `sub_index` is below that count. No box drops below 1
quantum by flow and no box changes sign. The amount is kept exactly. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.pair_rotation import pair_rotation
from physics.engine.pass_seed import pass_seed
from physics.engine.reach import reach
from physics.engine.settings import Settings
from physics.engine.state import State

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


@wp.kernel
def _hop(quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32), owner: wp.array3d(dtype=wp.int32),
         rates: wp.array2d(dtype=wp.float32), side: int, wrap: int, axis: int, parity: int, duration: float,
         seed: int, keys: wp.array(dtype=wp.int32), counts: wp.array(dtype=wp.int32), sub_index: int, pairs: int):
    ti, tj, tk = wp.tid()
    home = ti // side
    here = wp.vec3i(ti, tj, tk)
    if axis == 0:
        home = ti // pairs
        here[0] = home * side + parity + 2 * (ti - home * pairs)
    else:
        here[axis] = parity + 2 * here[axis]
    count = counts[home]
    if sub_index >= count:
        return
    step = wp.vec3i(0, 0, 0)
    step[axis] = 1
    there, valid = reach(here, step, home, side, wrap)
    if not valid:
        return
    near = quanta[here[0], here[1], here[2]]
    far = quanta[there[0], there[1], there[2]]
    if near == 0 or far == 0:
        return
    cell = owner[here[0], here[1], here[2]]
    if owner[there[0], there[1], there[2]] != cell:
        return
    tau = duration / float(count) * rates[home, cell]
    box_index = ((here[0] - home * side) * side + here[1]) * side + here[2]
    rng = wp.rand_init(seed ^ keys[home], box_index)
    moved, near_new, far_new = pair_rotation(wp.abs(near), wp.abs(far), phase[here[0], here[1], here[2]],
                                             phase[there[0], there[1], there[2]], -tau, wp.randf(rng))
    quanta[here[0], here[1], here[2]] = _signed(near, wp.abs(near) - moved)
    quanta[there[0], there[1], there[2]] = _signed(far, wp.abs(far) + moved)
    phase[here[0], here[1], here[2]] = _wrap(near_new)
    phase[there[0], there[1], there[2]] = _wrap(far_new)


def hop(state: State, settings: Settings, axis: int, parity: int, duration: float, pass_index: int,
        sub_index: int = 0) -> None:
    """Move whole quanta across the faces inside one cell along `axis` whose lower box has `parity`."""
    pairs = state.side // 2 if state.wrap else (state.side - parity) // 2
    if pairs <= 0:
        return
    dims = [state.side] * 3
    dims[0] = state.universes * state.side
    dims[axis] = pairs * (state.universes if axis == 0 else 1)
    seed = pass_seed(state.step, sub_index, pass_index)
    wp.launch(_hop, dim=tuple(dims), device=state.quanta.device,
              inputs=[state.quanta, state.phase, state.owner, state.cells_d["rate"], state.side, int(state.wrap),
                      int(axis), int(parity), float(duration), int(seed), state.keys, state.substeps, int(sub_index),
                      int(pairs)])
