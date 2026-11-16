"""Leak: energy passes from an energy volume box through one joint box to the next energy volume box, as one transfer.

Contract: `leak(state, settings, axis, group, duration, pass_index)` runs one pass of
the leak (bible rule 4: energy leaks from energy volume to energy volume only through
joints, A to B to A; plus and minus never cancel). Its pairs are two plus boxes two
apart along `axis` with a minus box between them (plus, minus, plus in a line): the
joint at its thinnest, one box across. The near box's coordinate along the axis, in
its own universe, is 4 m + 2 group or 4 m + 2 group + 1 (group 0 or 1), so the pairs of
one pass share no box and run together, and the two groups cover every pair. Each pair
takes one exact pair rotation (pair_rotation.py) with turn -t tau, t the joint box's
leak openness (leak_openness.py: graded by its squeeze, 0 past the seal) and tau =
duration * sqrt(rate_near rate_far) (local clocks); the whole quanta move straight
from one plus box to the other: they are never put into the joint box and never
change sign, and the joint box is only read. Two more local rules act on every leak
(bible rule 8, friction only in passing through a joint; drift from imperfect
transfer):
  friction  the phase step across the joint shrinks by kept = 1 / (1 + friction *
            (1 + s) * tau), s the joint box's squeeze (floored at 0): directed flow
            turned to jiggling; no quanta are made or lost
  spreading floor(spread_share * |moved| + u') quanta go from the more packed box of
            the pair to the less packed, at most half their difference: energy that
            crossed spreads a little; tallies[u, 0] counts them
A joint two or more boxes across lets nothing through until claims thin it to one box
(decision pending, rulefix decisions.md). The amount is exact and every box keeps its
sign. The rounding hash uses the box's place in its universe and the universe's key,
so a universe steps the same alone or side by side in a batch. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.clock_rate import clock_rate
from physics.engine.leak_openness import leak_openness
from physics.engine.pair_rotation import pair_rotation
from physics.engine.settings import Settings
from physics.engine.squeeze import squeeze
from physics.engine.state import State

PI = 3.14159265358979


@wp.func
def _wrap(angle: float) -> float:
    """The angle brought into (-pi, pi]."""
    return angle - 2.0 * PI * wp.floor((angle + PI) / (2.0 * PI))


@wp.kernel
def _leak(quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32), side: int, axis: int,
          group: int, duration: float, joint_rest: float, seal_squeeze: float, clock_packing: float,
          friction: float, spread_share: float, seed: int, keys: wp.array(dtype=wp.int32),
          tallies: wp.array2d(dtype=wp.int64)):
    ti, tj, tk = wp.tid()
    home = ti // side
    here = wp.vec3i(ti, tj, tk)
    local = here[axis]
    if axis == 0:
        local = ti - home * side
    if local % 4 // 2 != group or local + 2 >= side:
        return
    near = quanta[here[0], here[1], here[2]]
    if near <= 0:
        return
    middle_place = here
    middle_place[axis] = here[axis] + 1
    there = here
    there[axis] = here[axis] + 2
    middle = quanta[middle_place[0], middle_place[1], middle_place[2]]
    far = quanta[there[0], there[1], there[2]]
    if far <= 0:
        return
    open_share = leak_openness(middle, joint_rest, seal_squeeze)
    if open_share <= 0.0:
        return
    tau = duration * wp.sqrt(clock_rate(near, clock_packing) * clock_rate(far, clock_packing))
    box_index = ((here[0] - home * side) * side + here[1]) * side + here[2]
    rng = wp.rand_init(seed ^ keys[home], box_index)
    moved, near_new, far_new = pair_rotation(near, far, phase[here[0], here[1], here[2]],
                                             phase[there[0], there[1], there[2]], -open_share * tau, wp.randf(rng))
    near_amount = near - moved
    far_amount = far + moved
    joint_squeeze = wp.max(squeeze(middle, joint_rest), 0.0)
    kept = 1.0 / (1.0 + friction * (1.0 + joint_squeeze) * tau)
    rub = 0.5 * (1.0 - kept) * _wrap(far_new - near_new)
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
    quanta[here[0], here[1], here[2]] = near_amount
    quanta[there[0], there[1], there[2]] = far_amount
    phase[here[0], here[1], here[2]] = _wrap(near_new + rub)
    phase[there[0], there[1], there[2]] = _wrap(far_new - rub)


def leak(state: State, settings: Settings, axis: int, group: int, duration: float, pass_index: int) -> None:
    """Move whole plus quanta between plus boxes across one joint box along `axis`, pairs of `group` (0 or 1)."""
    seed = (state.step * 64 + pass_index) & 0x7FFFFFFF
    wp.launch(_leak, dim=state.shape, device=state.quanta.device,
              inputs=[state.quanta, state.phase, state.side, int(axis), int(group), float(duration),
                      settings.joint_rest, settings.seal_squeeze, settings.clock_packing, settings.friction,
                      settings.spread_share, int(seed), state.keys, state.tallies])
