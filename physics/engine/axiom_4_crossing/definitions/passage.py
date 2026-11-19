"""Passage: how freely plus energy passes a joint, read off the joint's own fluid.

A joint stands between two volumes along one direction, its axis. A box of the one volume
that faces the joint and the box of the other facing it (`facing.py`) are joined by the
joint's boxes once round its core there. `row_minus` is the minus quanta of that row, a box. Crossing energy rubs against them: `rub_share` is
the minus share of all the energy met in the passage. The two facing boxes move by the
fluid's own law, the pair rotation, with the turn cut to the share that does not rub
(`passing_turn`); friction then takes the rubbing share of the motion across, the step in
phase between the two boxes (`less_step`). No number is set here: the joint's quanta decide.

Stated a cell, not a box: the facing boxes are 1 in 6 `across` of their cell and a step is 1
in `across` squared of a swirl's turn, so the turn and the friction of one step are cut by
`layer_part`. The same share of a cell then crosses in a turn of its swirl at any boxes.
"""
import warp as wp

from physics.engine.axiom_1_energy.definitions.pair_rotation import TURN
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS
from physics.engine.axiom_2_swirls.definitions.ring_boxes import box_at


@wp.func
def row_minus(box_quanta: wp.array2d(dtype=wp.int32), slot: wp.int32, row: wp.int32,
              across: wp.int32, axis: wp.int32) -> wp.float64:
    """The minus quanta a box in the joint's outside boxes once round its core at row `row` (0 or less: no joint)."""
    held = wp.int32(0)
    for around in range(across):
        held -= box_quanta[slot, box_at(across, axis, across / 3 - 1, around, row % across)]
    return wp.float64(held) / wp.float64(across)


@wp.func
def rub_share(minus: wp.float64, low_has: wp.int32, high_has: wp.int32) -> wp.float64:
    """The share of crossing energy that rubs: minus over minus and plus, a box, in the passage."""
    plus = wp.float64(low_has + high_has) * wp.float64(0.5)
    return minus / (minus + plus)


@wp.func
def layer_part(across: wp.int32) -> wp.float64:
    """What one step at the facing boxes counts for: 1 at the least boxes across, less at more."""
    return wp.float64(ACROSS) / wp.float64(across)


@wp.func
def passing_turn(rub: wp.float64, across: wp.int32) -> wp.float64:
    """The turn of the two facing boxes' pair rotation: the fluid's own, less the rubbing share."""
    return wp.float64(TURN) * (wp.float64(1.0) - rub) * layer_part(across)


@wp.func
def less_step(low_phase: wp.float32, high_phase: wp.float32, rub: wp.float64) -> wp.vec2f:
    """The two facing phases with the rubbing share of the step between them taken, half each."""
    turn = float(wp.tau)
    step = low_phase - high_phase
    step -= turn * wp.floor((step + float(wp.pi)) / turn)  # the short way round
    low = low_phase - 0.5 * float(rub) * step
    high = high_phase + 0.5 * float(rub) * step
    return wp.vec2f(low - turn * wp.floor((low + float(wp.pi)) / turn),
                    high - turn * wp.floor((high + float(wp.pi)) / turn))
