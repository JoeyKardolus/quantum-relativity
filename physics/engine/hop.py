"""Hop: the quantum fluid moves whole quanta between face-neighbouring Planck boxes of one sign, one exact pair rotation each.

Contract: `hop(state, settings, axis, parity, duration, pass_index)` advances the fluid's
kinetic step (Schrodinger / Gross-Pitaevskii, hbar = box mass = 1:
i dpsi/dt = -1/2 sum over faces t_f (psi_there - psi) + on-site terms, on_site.py)
across every face along `axis` whose lower box has that coordinate's `parity`
in that box's own universe (a pair never spans two universes of a batch: the faces
between them are walls, no hop). Those faces share no box, so each pair runs on its
own: the scheme is De Raedt's product formula (1987), chosen over Succi's quantum
lattice Boltzmann and Visscher's staggered scheme because its pair rotation hands over
an exact amount across one face, which is what whole quanta need (pair_rotation.py):
the pair's turn is -t tau, tau = duration * sqrt(rate_a rate_b) (local clocks,
clock_rate.py), t the face's openness (openness.py: 1 between two boxes of one sign,
0 into empty space, across a plus-minus face, or off the grid: no hop; between plus
boxes of two different cells only when they are face-neighbouring energy volumes whose
shared joint (shared_joint.py) holds no quanta any more, state.cell_quanta: the joint
collapsed and the two volumes fell into each other; minus boxes of different joints
meet at seams and flow freely, the joints being one web); the rounding's
uniform comes from a hash of the box's place in its universe, the universe's key
(state.py), the step and the pass, so a universe steps the same alone or side by side
in a batch. With fluid substeps (substeps.py) universe u's pairs turn for duration /
state.substeps[u], only while `sub_index` is below that count, and the hash takes the
substep too. Inside an energy volume and inside the joint web the flow is frictionless
(bible rule 4). No box drops below 1 quantum by flow (a box empties only when it changes
cell, drain.py) and no box changes sign: quanta never cross between the kinds here; energy
passes between energy volumes only through a joint, as one transfer (leak.py).
The pair's amount is kept exactly, so sum |quanta| is exact over any number of steps.
Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.clock_rate import clock_rate
from physics.engine.openness import openness
from physics.engine.pair_rotation import pair_rotation
from physics.engine.pass_seed import pass_seed
from physics.engine.settings import Settings
from physics.engine.shared_joint import shared_joint
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
def _hop(quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32), side: int, axis: int,
         parity: int, duration: float, clock_packing: float, seed: int, keys: wp.array(dtype=wp.int32),
         counts: wp.array(dtype=wp.int32), sub_index: int, owner: wp.array3d(dtype=wp.int32),
         cell_quanta: wp.array2d(dtype=wp.int64), extent: wp.vec3i, lattice_counts: wp.array(dtype=wp.vec3i)):
    ti, tj, tk = wp.tid()
    pairs = (side - parity) // 2
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
    span = duration / float(count)
    there = here
    there[axis] = here[axis] + 1
    near = quanta[here[0], here[1], here[2]]
    far = quanta[there[0], there[1], there[2]]
    open_share = openness(near, far)
    if open_share <= 0.0:
        return
    own = owner[here[0], here[1], here[2]]
    other = owner[there[0], there[1], there[2]]
    if near > 0 and own != other:
        shared = shared_joint(own, other, extent, lattice_counts[home])
        if shared < 0:
            return
        if cell_quanta[home, shared] != wp.int64(0):
            return
    tau = span * wp.sqrt(clock_rate(near, clock_packing) * clock_rate(far, clock_packing))
    box_index = ((here[0] - home * side) * side + here[1]) * side + here[2]
    rng = wp.rand_init(seed ^ keys[home], box_index)
    moved, near_new, far_new = pair_rotation(wp.abs(near), wp.abs(far), phase[here[0], here[1], here[2]],
                                             phase[there[0], there[1], there[2]], -open_share * tau, wp.randf(rng))
    quanta[here[0], here[1], here[2]] = _signed(near, wp.abs(near) - moved)
    quanta[there[0], there[1], there[2]] = _signed(far, wp.abs(far) + moved)
    phase[here[0], here[1], here[2]] = _wrap(near_new)
    phase[there[0], there[1], there[2]] = _wrap(far_new)


def hop(state: State, settings: Settings, axis: int, parity: int, duration: float, pass_index: int,
        sub_index: int = 0) -> None:
    """Move whole quanta across the same-sign faces along `axis` whose lower box has `parity`, by exact pair rotations."""
    pairs = (state.side - parity) // 2
    if pairs <= 0:
        return
    dims = [state.side] * 3
    dims[0] = state.universes * state.side
    dims[axis] = pairs * (state.universes if axis == 0 else 1)
    seed = pass_seed(state.step, sub_index, pass_index)
    wp.launch(_hop, dim=tuple(dims), device=state.quanta.device,
              inputs=[state.quanta, state.phase, state.side, int(axis), int(parity), float(duration),
                      settings.clock_packing, int(seed), state.keys, state.substeps, int(sub_index),
                      state.owner, state.cell_quanta, wp.vec3i(*state.extent), state.counts])
