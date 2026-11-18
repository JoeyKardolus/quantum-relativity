"""Flow inside: the fluid of a cell moves between the boxes of its own block.

Contract: `flow_inside(state, settings, axis, parity)` takes one step of every pair of
boxes that share a face along `axis` (0, 1, 2) inside one cell's block: the pairs
`(i, i + 1)` with `i` in 0, 2, 4 for parity 0 and in 1, 3 for parity 1, so no box is in
two pairs of one call and no pair leaves its block. The step's six calls (three axes,
two parities) take every face inside the block once.
For one pair, with `tau = settings.step_time * state.rate[cell]` (the cell's local time):
  1. the two boxes' waves `sqrt(quanta) e^{i phase}` turn into each other by
     `pair_rotation` with turn `tau`: fluid flows from the wave whose phase is ahead;
  2. the flow becomes whole quanta by `whole_move`: `floor(flow + uniform)`, the uniform
     from `draw_uniform` with pass `FLOW_ROUND + axis * 2 + parity` and the index of the
     pair's lower box, cut so that neither box goes past 0;
  3. the whole number is taken from one box and added to the other, so the cell's total
     and the sign of its boxes never change; both boxes get the rotation's phases,
     wrapped into (-pi, pi].
Nothing else drives the fluid inside a cell: there is no pressure, so no plus energy
pushes plus energy. What is uneven evens out and swings back only as waves do.
Writes `quanta` and `phase` of the two boxes, `moved_inside[cell]` (quanta moved, added
up) and `cut_moves[cell]` (moves that were cut), and nothing else. A pair of two empty
boxes does nothing and an id that is no cell is skipped.
Reads `quanta`, `phase`, `rate`, `kind`. No thread reads a box of another cell.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform
from physics.engine.axiom_1_energy.definitions.pair_rotation import pair_rotation
from physics.engine.axiom_1_energy.definitions.passes import FLOW_ROUND
from physics.engine.axiom_1_energy.definitions.whole_move import whole_move
from physics.engine.axiom_7_quantum.definitions.wrap_phase import wrap_phase
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, BOXES, NO_CELL

wp.set_module_options({"fuse_fp": False})

_NO_CELL = wp.constant(NO_CELL)
_BLOCK = wp.constant(BLOCK)
_BOXES = wp.constant(wp.uint64(BOXES))
_AXES = 3
_WORD = 0xFFFFFFFF


@wp.func
def _lower_box(axis: wp.int32, low: wp.int32, first: wp.int32, second: wp.int32) -> wp.vec3i:
    """The lower box of a pair: `low` along `axis`, `first` and `second` across it in rising order."""
    if axis == 0:
        return wp.vec3i(low, first, second)
    if axis == 1:
        return wp.vec3i(first, low, second)
    return wp.vec3i(first, second, low)


@wp.func
def _step_along(axis: wp.int32) -> wp.vec3i:
    """One box further along `axis`."""
    if axis == 0:
        return wp.vec3i(1, 0, 0)
    if axis == 1:
        return wp.vec3i(0, 1, 0)
    return wp.vec3i(0, 0, 1)


@wp.kernel
def _flow(kind: wp.array(dtype=wp.int32), rate: wp.array(dtype=wp.float32), axis: wp.int32, parity: wp.int32,
          step_time: wp.float64, seed: wp.uint32, step: wp.uint32, pass_id: wp.uint32,
          quanta: wp.array4d(dtype=wp.int32), phase: wp.array4d(dtype=wp.float64),
          moved_inside: wp.array(dtype=wp.int64), cut_moves: wp.array(dtype=wp.int32)):
    cell, pair, first, second = wp.tid()
    sign = kind[cell]
    if sign == _NO_CELL:
        return
    lower = _lower_box(axis, pair * 2 + parity, first, second)
    upper = lower + _step_along(axis)
    lower_has = wp.int64(wp.abs(quanta[cell, lower[0], lower[1], lower[2]]))
    upper_has = wp.int64(wp.abs(quanta[cell, upper[0], upper[1], upper[2]]))
    if lower_has == wp.int64(0) and upper_has == wp.int64(0):
        return
    local_time = step_time * wp.float64(rate[cell])
    lower_amount, lower_phase, upper_amount, upper_phase = pair_rotation(
        wp.float64(lower_has), wp.float64(phase[cell, lower[0], lower[1], lower[2]]),
        wp.float64(upper_has), wp.float64(phase[cell, upper[0], upper[1], upper[2]]), local_time)
    flow = upper_amount - wp.float64(upper_has)
    box = wp.uint64(cell) * _BOXES + wp.uint64((lower[0] * _BLOCK + lower[1]) * _BLOCK + lower[2])
    uniform = draw_uniform(seed, step, pass_id, box)
    moved = whole_move(flow, uniform, lower_has, upper_has)
    if moved != wp.int64(wp.floor(flow + uniform)):
        wp.atomic_add(cut_moves, cell, 1)
    quanta[cell, lower[0], lower[1], lower[2]] = sign * wp.int32(lower_has - moved)
    quanta[cell, upper[0], upper[1], upper[2]] = sign * wp.int32(upper_has + moved)
    phase[cell, lower[0], lower[1], lower[2]] = wp.float64(wrap_phase(lower_phase))
    phase[cell, upper[0], upper[1], upper[2]] = wp.float64(wrap_phase(upper_phase))
    wp.atomic_add(moved_inside, cell, wp.abs(moved))


def flow_inside(state, settings, axis: int, parity: int) -> None:
    """One step of every pair of boxes across a face along `axis`, of one parity, inside each cell."""
    if axis not in range(_AXES) or parity not in (0, 1):
        raise ValueError(f"axis must be 0, 1 or 2 and parity 0 or 1, got {axis} and {parity}")
    pairs = (BLOCK - parity) // 2
    wp.launch(_flow, dim=(state.count, pairs, BLOCK, BLOCK), device=state.device,
              inputs=[state.kind, state.rate, axis, parity, wp.float64(settings.step_time),
                      wp.uint32(state.seed & _WORD), wp.uint32(state.step & _WORD), wp.uint32(FLOW_ROUND + axis * 2 + parity),
                      state.quanta, state.phase, state.moved_inside, state.cut_moves])
