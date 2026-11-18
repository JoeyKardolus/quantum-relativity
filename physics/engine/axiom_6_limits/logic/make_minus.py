"""Make minus: a squeezed joint makes more minus energy out of itself.

Contract: `make_minus(state, settings)` runs one thread per joint. A joint whose
`squeeze` is above 1 makes `floor(make_law + uniform)` whole minus quanta (`MAKE_ROUND`,
the joint's id as index) in `tau = step_time * rate[joint]` of its own time, and puts
them into its own boxes in proportion to what each holds; what is left over after the
whole shares goes one quantum each to the boxes that hold something, from box 0 up. A
box holding nothing gets nothing. `made[joint]` is the number made, and `made_total`
(axiom 1's books) rises by the sum. Phases are untouched. Nothing else changes a living
joint's quanta, so they never fall.
Reads `kind`, `squeeze`, `cell_quanta`, `rate`. No thread reads a box of another cell.
A making that would put more than 2 ** 31 - 1 quanta in one box makes nothing and raises
OverflowError: nothing wraps silently.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.axiom_1_energy.definitions.cell_sum import cell_sum
from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform
from physics.engine.axiom_1_energy.definitions.passes import MAKE_ROUND
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, JOINT
from physics.engine.axiom_6_limits.definitions.make_law import make_law

wp.set_module_options({"fuse_fp": False})

_BLOCK = wp.constant(BLOCK)
_MOST_IN_A_BOX = wp.constant(wp.float64(2 ** 31 - 1))


@wp.func
def _deal_shares(quanta: wp.array4d(dtype=wp.int32), cell: wp.int32, count: wp.int64, holds: wp.int64) -> wp.int64:
    """Give every box its whole share of `count` by what it holds; returns what is left over."""
    dealt = wp.int64(0)
    for across in range(_BLOCK):
        for along in range(_BLOCK):
            for up in range(_BLOCK):
                share = count * wp.int64(-quanta[cell, across, along, up]) / holds
                quanta[cell, across, along, up] = quanta[cell, across, along, up] - wp.int32(share)
                dealt += share
    return count - dealt


@wp.func
def _fullest_box(quanta: wp.array4d(dtype=wp.int32), cell: wp.int32) -> wp.int64:
    """The most quanta any one box of a joint holds, ignoring sign."""
    most = wp.int64(0)
    for across in range(_BLOCK):
        for along in range(_BLOCK):
            for up in range(_BLOCK):
                most = wp.max(most, wp.int64(-quanta[cell, across, along, up]))
    return most


@wp.func
def _deal_rest(quanta: wp.array4d(dtype=wp.int32), cell: wp.int32, rest: wp.int64):
    """Hand the leftover quanta out one each to the boxes that hold something, from box 0 up."""
    left = rest
    for across in range(_BLOCK):
        for along in range(_BLOCK):
            for up in range(_BLOCK):
                if left > wp.int64(0) and quanta[cell, across, along, up] < 0:
                    quanta[cell, across, along, up] = quanta[cell, across, along, up] - 1
                    left -= wp.int64(1)


@wp.kernel
def _make_in_joint(kind: wp.array(dtype=wp.int32), squeeze: wp.array(dtype=wp.float32),
                   cell_quanta: wp.array(dtype=wp.int64), rate: wp.array(dtype=wp.float32),
                   quanta: wp.array4d(dtype=wp.int32), made: wp.array(dtype=wp.int64), seed: wp.uint32,
                   step: wp.uint32, step_time: wp.float64, squeeze_rate: wp.float64):
    """One thread per cell id: a squeezed joint adds whole minus quanta to its own boxes."""
    cell = wp.tid()
    if kind[cell] != JOINT:
        return
    tau = step_time * wp.float64(rate[cell])
    wanted = make_law(cell_quanta[cell], wp.float64(squeeze[cell]), squeeze_rate, tau)
    if not (wanted > wp.float64(0.0)):
        return
    count = wp.int64(wp.floor(wanted + draw_uniform(seed, step, wp.uint32(MAKE_ROUND), wp.uint64(cell))))
    holds = cell_sum(quanta, cell)
    if count <= wp.int64(0) or holds <= wp.int64(0):
        return
    most = _fullest_box(quanta, cell)
    if wp.float64(most) * (wp.float64(1.0) + wp.float64(count) / wp.float64(holds)) + wp.float64(1.0) > _MOST_IN_A_BOX:
        made[cell] = wp.int64(-1)
        return
    _deal_rest(quanta, cell, _deal_shares(quanta, cell, count, holds))
    made[cell] = count


def make_minus(state, settings) -> None:
    """Let every squeezed joint make its whole minus quanta for this step, booked in `made_total`."""
    wp.launch(_make_in_joint, dim=state.count,
              inputs=[state.kind, state.squeeze, state.cell_quanta, state.rate, state.quanta, state.made,
                      wp.uint32(state.seed), wp.uint32(state.step), wp.float64(settings.step_time),
                      wp.float64(settings.squeeze_rate)],
              device=state.device)
    made = state.made.numpy()[:state.count]
    if (made < 0).any():
        raise OverflowError(f"joint {int(np.flatnonzero(made < 0)[0])} would make more than a box can hold")
    state.made_total += int(made.sum())
