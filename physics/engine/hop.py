"""Hop: the quantum fluid moves whole quanta between face-neighbouring Planck boxes, one exact pair rotation each.

Contract: `hop(state, settings, axis, parity, duration, pass_index)` advances the fluid's
kinetic step (Schrodinger / Gross-Pitaevskii, hbar = box mass = 1:
i dpsi/dt = -1/2 sum over faces t_f (psi_there - psi) + on-site terms, on_site.py)
across every face along `axis` whose lower box has that coordinate's `parity`
in that box's own universe (a pair never spans two universes of a batch: the faces
between them are walls, no hop). Those
faces share no box, so each pair runs on its own: the scheme is De Raedt's product
formula (1987), chosen over Succi's quantum lattice Boltzmann and Visscher's staggered
scheme because its pair rotation hands over an exact amount across one face, which is
what whole quanta need:
  1. the pair's complex numbers are rebuilt from the state, a = sqrt(|n_a|) e^{i phase_a};
  2. the pair's own equation H = t/2 [[1, -1], [-1, 1]] is solved exactly: the mean
     stays and the half difference turns by exp(-i t tau), tau = duration *
     sqrt(rate_a rate_b) (local clocks, clock_rate.py), t the face's openness
     (openness.py; 0 into empty space, past the seal, or off the grid: no hop);
  3. the rotation moves flux = |a|^2 - |a'|^2 = 2 Re(conj(mean) (half - half')) from a
     to b, a real number; the quanta moved are floor(flux + u), u uniform in [0, 1)
     from a hash of the box's place in its universe, the universe's key (state.py),
     the step and the pass, so the rounding is unbiased, needs no carry, and a
     universe steps the same alone or side by side in a batch; clamped so neither box drops below 1 quantum (a box empties only by a
     claim, claim.py, never by flow);
  4. each box keeps the new phase arg(a') and its size is sqrt of its new whole quanta.
The pair's amount is kept exactly, so sum |quanta| is exact over any number of steps;
the fluid is unitary up to that rounding of at most one quantum per face per pass.
Each box keeps its sign: quanta that cross a plus-minus face join the receiving box's
kind.
On a crossing face (one plus box, one minus box) two more local rules act (bible rule 8,
friction only in passing through a joint; drift from imperfect transfer):
  friction  the phase step across the face shrinks by kept = 1 / (1 + friction *
            (1 + s) * tau), s the joint box's squeeze (floored at 0): directed flow
            turned to jiggling; no quanta are made or lost
  spreading floor(spread_share * |moved| + u') quanta go from the more packed box of
            the pair to the less packed, at most half their difference: energy that
            crossed spreads a little and gets less concentrated; tallies[u, 0] counts them
Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.clock_rate import clock_rate
from physics.engine.openness import openness
from physics.engine.settings import Settings
from physics.engine.squeeze import squeeze
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
         parity: int, duration: float, joint_rest: float, seal_squeeze: float, clock_packing: float,
         friction: float, spread_share: float, seed: int, keys: wp.array(dtype=wp.int32),
         tallies: wp.array2d(dtype=wp.int64)):
    ti, tj, tk = wp.tid()
    pairs = (side - parity) // 2
    home = ti // side
    here = wp.vec3i(ti, tj, tk)
    if axis == 0:
        home = ti // pairs
        here[0] = home * side + parity + 2 * (ti - home * pairs)
    else:
        here[axis] = parity + 2 * here[axis]
    there = here
    there[axis] = here[axis] + 1
    near = quanta[here[0], here[1], here[2]]
    far = quanta[there[0], there[1], there[2]]
    open_share = openness(near, far, joint_rest, seal_squeeze)
    if open_share <= 0.0:
        return
    tau = duration * wp.sqrt(clock_rate(near, clock_packing) * clock_rate(far, clock_packing))
    near_size = float(wp.abs(near))
    far_size = float(wp.abs(far))
    near_angle = phase[here[0], here[1], here[2]]
    far_angle = phase[there[0], there[1], there[2]]
    near_psi = wp.vec2(wp.sqrt(near_size) * wp.cos(near_angle), wp.sqrt(near_size) * wp.sin(near_angle))
    far_psi = wp.vec2(wp.sqrt(far_size) * wp.cos(far_angle), wp.sqrt(far_size) * wp.sin(far_angle))
    mean = (near_psi + far_psi) * 0.5
    half = (near_psi - far_psi) * 0.5
    turn = -open_share * tau
    turned = wp.vec2(half[0] * wp.cos(turn) - half[1] * wp.sin(turn), half[0] * wp.sin(turn) + half[1] * wp.cos(turn))
    change = half - turned
    flux = 2.0 * (mean[0] * change[0] + mean[1] * change[1])
    box_index = ((here[0] - home * side) * side + here[1]) * side + here[2]
    rng = wp.rand_init(seed ^ keys[home], box_index)
    moved = int(wp.floor(flux + wp.randf(rng)))
    near_amount = wp.abs(near)
    far_amount = wp.abs(far)
    moved = wp.min(moved, near_amount - 1)
    moved = wp.max(moved, -(far_amount - 1))
    near_amount = near_amount - moved
    far_amount = far_amount + moved
    near_after = mean + turned
    far_after = mean - turned
    near_new = wp.atan2(near_after[1], near_after[0])
    far_new = wp.atan2(far_after[1], far_after[0])
    if (near > 0) != (far > 0):
        joint_squeeze = wp.max(squeeze(wp.min(near, far), joint_rest), 0.0)
        kept = 1.0 / (1.0 + friction * (1.0 + joint_squeeze) * tau)
        rub = 0.5 * (1.0 - kept) * _wrap(far_new - near_new)
        near_new = near_new + rub
        far_new = far_new - rub
        spread = int(wp.floor(spread_share * float(wp.abs(moved)) + wp.randf(rng)))
        if near_amount > far_amount:
            spread = wp.min(spread, (near_amount - far_amount) // 2)
            near_amount = near_amount - spread
            far_amount = far_amount + spread
        else:
            spread = wp.min(spread, (far_amount - near_amount) // 2)
            far_amount = far_amount - spread
            near_amount = near_amount + spread
        if spread > 0:
            wp.atomic_add(tallies, home, 0, wp.int64(spread))
    quanta[here[0], here[1], here[2]] = _signed(near, near_amount)
    quanta[there[0], there[1], there[2]] = _signed(far, far_amount)
    phase[here[0], here[1], here[2]] = _wrap(near_new)
    phase[there[0], there[1], there[2]] = _wrap(far_new)


def hop(state: State, settings: Settings, axis: int, parity: int, duration: float, pass_index: int) -> None:
    """Move whole quanta across the faces along `axis` whose lower box has `parity`, by exact pair rotations."""
    pairs = (state.side - parity) // 2
    if pairs <= 0:
        return
    dims = [state.side] * 3
    dims[0] = state.universes * state.side
    dims[axis] = pairs * (state.universes if axis == 0 else 1)
    seed = (state.step * 64 + pass_index) & 0x7FFFFFFF
    wp.launch(_hop, dim=tuple(dims), device=state.quanta.device,
              inputs=[state.quanta, state.phase, state.side, int(axis), int(parity),
                      float(duration), settings.joint_rest, settings.seal_squeeze, settings.clock_packing,
                      settings.friction, settings.spread_share, int(seed), state.keys, state.tallies])
