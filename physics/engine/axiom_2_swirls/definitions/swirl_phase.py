"""Swirl phase: a cell is a swirl because its fluid's phase goes once round its axis.

Contract: `swirl_phase(axis, sense, across, along, up)` (Warp func; int32s) returns as float64 the
phase, in (-pi, pi], of box (across, along, up) of a cell that is one whole swirl round `axis` in
the sense `sense` (+1 or -1): `-sense` times the angle of the box round the line through
the middle of the block (between the boxes, at 2.5, 2.5), measured from the first
coordinate across the axis toward the second (axis 0: along toward up, axis 1: across toward up,
axis 2: across toward along). A box on the outside of the block has the phase of the box behind
it (its coordinates across the axis are brought into 1 to 4 first), so the phase stands
level from the inner boxes out to the block's sides: the swirl carries fluid round the
axis and none toward a side. The same in every layer along the axis.
`swirl_phase_host` is the host twin.
Fluid flows from the wave whose phase is ahead (pair_rotation.py), so with the phase
falling round the axis the cell's own fluid goes round it the way `sense` says: the
swirl is the fluid's own motion, not a rule that moves it. Going once round the axis
the phase comes back to itself after exactly one whole turn; a swirl can be a whole
number of turns and nothing between (axiom 2: "in whole steps"), and this is one.
This is book-keeping inside one cell: it gives no box a place in space.
"""
from __future__ import annotations

import math

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import BLOCK

wp.set_module_options({"fuse_fp": False})

_MIDDLE = (BLOCK - 1) / 2.0
_MIDDLE_W = wp.constant(wp.float64(_MIDDLE))
_INNER_LOW = 1
_INNER_HIGH = BLOCK - 2


@wp.func
def swirl_phase(axis: wp.int32, sense: wp.int32, across: wp.int32, along: wp.int32, up: wp.int32) -> wp.float64:
    """The phase of box (across, along, up) in a cell that is one whole swirl round `axis`."""
    first = wp.float64(wp.clamp(across, _INNER_LOW, _INNER_HIGH))
    second = wp.float64(wp.clamp(along, _INNER_LOW, _INNER_HIGH))
    if axis == 0:
        first = wp.float64(wp.clamp(along, _INNER_LOW, _INNER_HIGH))
        second = wp.float64(wp.clamp(up, _INNER_LOW, _INNER_HIGH))
    elif axis == 1:
        second = wp.float64(wp.clamp(up, _INNER_LOW, _INNER_HIGH))
    return -wp.float64(sense) * wp.atan2(second - _MIDDLE_W, first - _MIDDLE_W)


def swirl_phase_host(axis: int, sense: int, across: int, along: int, up: int) -> float:
    """The phase of box (across, along, up) in a cell that is one whole swirl round `axis` (host)."""
    first, second = (min(max(int(value), _INNER_LOW), _INNER_HIGH) for value in ((along, up), (across, up), (across, along))[int(axis)])
    return -int(sense) * math.atan2(second - _MIDDLE, first - _MIDDLE)
