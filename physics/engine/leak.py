"""Leak: energy passes from an energy volume box through one joint to the next energy volume box, as one transfer.

Contract: `leak(state, settings, axis, group, duration, pass_index, sub_index=0)` runs
one pass of the leak (bible rule 4: energy leaks from energy volume to energy volume
only through joints, A to B to A; plus and minus never cancel). Owner, 2026-11-16:
counted in cells there is exactly one joint between two energy volumes, counted in
Planck boxes a joint is as thick as its energy makes it, and energy leaks from one
energy volume to the next through exactly one joint cell. So a crossing is a straight
line along `axis`: a plus box, one or more minus boxes (the joint, any thickness), a
plus box (plus, minus ... minus, plus); an empty box or the wall breaks the line. Which
joint a crossing passes is decided by the forced cell lattice, never guessed from the
boxes (owner, 2026-11-16): a line is a crossing only when its two plus boxes belong to
two face-neighbouring energy volumes and every minus box on it belongs to the one joint
they share (shared_joint.py); any other line (through another joint, or between volumes
that are not face neighbours) carries nothing. A state without a cell lattice
(state.cells_per_side 0) leaks nothing. One
thread walks each line of its universe and counts the crossings on it from the low
end; it takes the crossings whose count has parity `group` (0 or 1), so the crossings
of one pass share no box and the two groups cover them all. Each crossing takes one
exact pair rotation (pair_rotation.py) between its two plus boxes with turn -t tau, t
the joint's leak openness (leak_openness.py: graded by the joint's packing along the
crossing, the mean |quanta| of its minus boxes; 0 past the seal) and tau = span *
sqrt(rate_near rate_far) (local clocks), span = duration / state.substeps[u] (fluid
substeps, substeps.py; a universe whose count is at or below `sub_index` is skipped).
The whole quanta move straight from one plus box to the other: they are never put into
the joint and never change sign, and the joint's boxes are only read. Two more local
rules act on every crossing (bible rule 8, friction only in passing through a joint;
drift from imperfect transfer):
  friction  the phase step across the joint shrinks by kept = 1 / (1 + friction *
            (1 + s) * tau), s the joint's squeeze along the crossing (floored at 0):
            directed flow turned to jiggling; no quanta are made or lost
  spreading floor(spread_share * |moved| + u') quanta go from the more packed box of
            the pair to the less packed, at most half their difference: energy that
            crossed spreads a little; tallies[u, 0] counts them
The amount is exact and every box keeps its sign. The rounding hash uses the near box's
place in its universe, the universe's key and pass_seed(step, sub_index, pass_index),
so a universe steps the same alone or side by side in a batch. Returns None.
"""
from __future__ import annotations

import warp as wp

from physics.engine.clock_rate import clock_rate
from physics.engine.leak_openness import leak_openness
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
def _place(universe: int, side: int, axis: int, first: int, second: int, along: int) -> wp.vec3i:
    """The box at `along` on the line along `axis` through (first, second) of universe `universe`."""
    if axis == 0:
        return wp.vec3i(universe * side + along, first, second)
    if axis == 1:
        return wp.vec3i(universe * side + first, along, second)
    return wp.vec3i(universe * side + first, second, along)


@wp.func
def _cross(quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32), here: wp.vec3i,
           there: wp.vec3i, packing: float, span: float, home: int, side: int, joint_rest: float,
           seal_squeeze: float, clock_packing: float, friction: float, spread_share: float, seed: int,
           keys: wp.array(dtype=wp.int32), tallies: wp.array2d(dtype=wp.int64)):
    """One crossing: an exact pair rotation between the two plus boxes, with friction and spreading."""
    open_share = leak_openness(packing, joint_rest, seal_squeeze)
    if open_share <= 0.0:
        return
    near = quanta[here[0], here[1], here[2]]
    far = quanta[there[0], there[1], there[2]]
    tau = span * wp.sqrt(clock_rate(near, clock_packing) * clock_rate(far, clock_packing))
    box_index = ((here[0] - home * side) * side + here[1]) * side + here[2]
    rng = wp.rand_init(seed ^ keys[home], box_index)
    moved, near_new, far_new = pair_rotation(near, far, phase[here[0], here[1], here[2]],
                                             phase[there[0], there[1], there[2]], -open_share * tau, wp.randf(rng))
    near_amount = near - moved
    far_amount = far + moved
    joint_squeeze = wp.max(packing / joint_rest - 1.0, 0.0)
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


@wp.kernel
def _leak(quanta: wp.array3d(dtype=wp.int32), phase: wp.array3d(dtype=wp.float32), side: int, axis: int,
          group: int, duration: float, joint_rest: float, seal_squeeze: float, clock_packing: float,
          friction: float, spread_share: float, seed: int, keys: wp.array(dtype=wp.int32),
          tallies: wp.array2d(dtype=wp.int64), counts: wp.array(dtype=wp.int32), sub_index: int,
          owner: wp.array3d(dtype=wp.int32), cells_per_side: int):
    home, first, second = wp.tid()
    count = counts[home]
    if sub_index >= count:
        return
    span = duration / float(count)
    last_plus = int(-1)
    joint_boxes = int(0)
    joint_quanta = float(0.0)
    joint_cell = int(-1)
    crossings = int(0)
    for along in range(side):
        place = _place(home, side, axis, first, second, along)
        held = quanta[place[0], place[1], place[2]]
        cell = owner[place[0], place[1], place[2]]
        if held == 0:
            last_plus = -1
            joint_boxes = 0
            joint_quanta = 0.0
        elif held < 0:
            if last_plus >= 0:
                if joint_boxes == 0:
                    joint_cell = cell
                elif joint_cell != cell:
                    joint_cell = -2
                joint_boxes += 1
                joint_quanta += float(-held)
        else:
            near = _place(home, side, axis, first, second, wp.max(last_plus, 0))
            shared = shared_joint(owner[near[0], near[1], near[2]], cell, cells_per_side)
            if last_plus >= 0 and joint_boxes > 0 and shared >= 0 and joint_cell == shared:
                if crossings % 2 == group:
                    _cross(quanta, phase, _place(home, side, axis, first, second, last_plus), place,
                           joint_quanta / float(joint_boxes), span, home, side, joint_rest, seal_squeeze,
                           clock_packing, friction, spread_share, seed, keys, tallies)
                crossings += 1
            last_plus = along
            joint_boxes = 0
            joint_quanta = 0.0


def leak(state: State, settings: Settings, axis: int, group: int, duration: float, pass_index: int,
         sub_index: int = 0) -> None:
    """Move whole plus quanta between plus boxes across their shared joint along `axis`, the crossings of `group`."""
    if state.cells_per_side == 0:
        return
    seed = pass_seed(state.step, sub_index, pass_index)
    wp.launch(_leak, dim=(state.universes, state.side, state.side), device=state.quanta.device,
              inputs=[state.quanta, state.phase, state.side, int(axis), int(group), float(duration),
                      settings.joint_rest, settings.seal_squeeze, settings.clock_packing, settings.friction,
                      settings.spread_share, int(seed), state.keys, state.tallies, state.substeps, int(sub_index),
                      state.owner, state.cells_per_side])
