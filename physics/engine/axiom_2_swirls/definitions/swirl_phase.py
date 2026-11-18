"""Swirl phase: a cell is a swirl because its fluid's phase goes once round its axis.

Contract: `swirl_phase(axis, sense, turned, across, along, up)` (Warp func; int32s)
returns as float64 the phase, in (-pi, pi], of box (across, along, up) of a cell that is
one whole standing swirl round `axis` in the sense `sense` (+1 or -1). With `first` and
`second` the box's two coordinates across the axis (axis 0: along and up, axis 1: across
and up, axis 2: across and along) and `m(c) = sin(pi (c - 2.5) / 6)`, the cell's wave is
`m(first) - i sense m(second)` (times -1 when `turned` is 1: half a turn on), `swirl_share`
its size squared, and this its angle. The same in every layer along the axis.
`swirl_phase_host` is the host twin.
Why this wave: `m` is the slowest standing wave of a row of six boxes, so each of the
two parts only turns in place and the swirl stands as it is under the fluid's own step
(`flow_inside`); and at a side of the block `m` is +-sin(75 degrees), so two neighbour
cells that mirror each other (opposite sense across a face the axis lies in, `turned`
across the first coordinate) hold equal waves in the boxes that face each other. Equal
waves stand still (pair_rotation.py): a grid of such swirls meshes like gears and sends
nothing through a joint by itself.
Fluid flows from the wave whose phase is ahead, so with the phase falling round the axis
the cell's own fluid goes round it the way `sense` says. Going once round the axis the
phase comes back to itself after exactly one whole turn (axiom 2: "in whole steps").
This is book-keeping inside one cell: it gives no box a place in space.
"""
from __future__ import annotations

import math

import warp as wp

from physics.engine.axiom_3_structure.definitions.kinds import BLOCK

wp.set_module_options({"fuse_fp": False})

_MIDDLE = (BLOCK - 1) / 2.0
_MIDDLE_W = wp.constant(wp.float64(_MIDDLE))
_RATE_W = wp.constant(wp.float64(math.pi / BLOCK))
_PI = wp.constant(wp.float64(math.pi))


@wp.func
def swirl_mode(coordinate: wp.int32) -> wp.float64:
    """The slowest standing wave of a row of six boxes, at one of them."""
    return wp.sin(_RATE_W * (wp.float64(coordinate) - _MIDDLE_W))


@wp.func
def swirl_phase(axis: wp.int32, sense: wp.int32, turned: wp.int32, across: wp.int32, along: wp.int32,
                up: wp.int32) -> wp.float64:
    """The phase of box (across, along, up) in a cell that is one whole standing swirl round `axis`."""
    first = swirl_mode(across)
    second = swirl_mode(along)
    if axis == 0:
        first = swirl_mode(along)
        second = swirl_mode(up)
    elif axis == 1:
        second = swirl_mode(up)
    if turned == 1:
        return wp.atan2(wp.float64(sense) * second, -first)
    return wp.atan2(-wp.float64(sense) * second, first)


def swirl_mode_host(coordinate: int) -> float:
    """The slowest standing wave of a row of six boxes, at one of them (host)."""
    return math.sin(math.pi / BLOCK * (int(coordinate) - _MIDDLE))


def swirl_phase_host(axis: int, sense: int, turned: int, across: int, along: int, up: int) -> float:
    """The phase of box (across, along, up) in a cell that is one whole standing swirl round `axis` (host)."""
    first, second = (swirl_mode_host(value) for value in ((along, up), (across, up), (across, along))[int(axis)])
    if int(turned) == 1:
        return math.atan2(int(sense) * second, -first)
    return math.atan2(-int(sense) * second, first)
