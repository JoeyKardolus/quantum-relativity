"""Start swirls: every cell of the start, energy volume and joint, begins as a standing swirl.

Contract: `start_swirls(state)` runs once on a fresh start lattice (step 0, before any
event; ValueError otherwise). It draws from the seed one axis for the whole grid
(`START_SWIRL_AXIS`, index 0: 0, 1 or 2, each as likely) and which half of the cells
goes round which way (`START_SWIRL_SENSE`, index 0), and gives every living cell
  swirl_axis   that axis
  swirl_sense  +1 or -1, opposite for two cells that are neighbours across the axis'
               first or second coordinate (the cell's place in the start lattice,
               counted in volumes: every second index, as the lattice is laid)
and writes the swirl into the cell's own fluid: every box gets the phase `swirl_phase`
gives it (`turned` for every second cell along the first coordinate), and the cell's
quanta are dealt again over its boxes in proportion to what each held times
`swirl_share`, whole and exactly (whole parts, the rest one each from box 0 up), so
the cell holds what it held and its core stands open.
Neighbours mirror each other, like meshing gears: the boxes of two energy volumes that
face each other across a joint hold equal waves, so the swirls stand and nothing crosses
a joint by itself. From then on nothing turns a swirl: the fluid goes round by its own
phase (`flow_inside`, axiom 1), on the cell's local time. An id that is no cell is left
alone. Reads `kind`, `seed`, `quanta`. Writes `swirl_axis`, `swirl_sense`, `phase` and
the boxes' `quanta`; no quantum leaves its cell.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform_host
from physics.engine.axiom_1_energy.definitions.passes import START_SWIRL_AXIS, START_SWIRL_SENSE
from physics.engine.axiom_2_swirls.definitions.swirl_phase import swirl_phase
from physics.engine.axiom_2_swirls.definitions.swirl_share import swirl_share
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, NO_CELL

wp.set_module_options({"fuse_fp": False})

_NO_CELL = wp.constant(wp.int32(NO_CELL))
_BLOCK = wp.constant(BLOCK)
_AXES = 3


@wp.func
def _across_parity(cell: wp.int32, side: wp.int32, axis: wp.int32) -> wp.vec2i:
    """Whether a cell of the start lattice is an odd volume along the first and the second coordinate across the axis."""
    place = wp.vec3i(cell / (side * side), (cell / side) % side, cell % side)
    first = place[0]
    second = place[1]
    if axis == 0:
        first = place[1]
        second = place[2]
    elif axis == 1:
        second = place[2]
    return wp.vec2i((first / 2) % 2, (second / 2) % 2)


@wp.kernel
def _wind(kind: wp.array(dtype=wp.int32), side: wp.int32, axis: wp.int32, flip: wp.int32,
          swirl_axis: wp.array(dtype=wp.int32), swirl_sense: wp.array(dtype=wp.int32),
          quanta: wp.array4d(dtype=wp.int32), phase: wp.array4d(dtype=wp.float64)):
    """One thread per cell: its sense by its place, the swirl's phase in every box, its quanta dealt as a swirl holds them."""
    cell = wp.tid()
    if kind[cell] == _NO_CELL:
        return
    odd = _across_parity(cell, side, axis)
    sense = flip
    if (odd[0] + odd[1]) % 2 == 1:
        sense = -flip
    swirl_axis[cell] = axis
    swirl_sense[cell] = sense
    held = wp.int64(0)
    weight = wp.float64(0.0)
    sign = 1
    for across in range(_BLOCK):
        for along in range(_BLOCK):
            for up in range(_BLOCK):
                has = quanta[cell, across, along, up]
                if has < 0:
                    sign = -1
                held += wp.int64(wp.abs(has))
                weight += wp.float64(wp.abs(has)) * swirl_share(axis, across, along, up)
    left = held
    for across in range(_BLOCK):
        for along in range(_BLOCK):
            for up in range(_BLOCK):
                phase[cell, across, along, up] = swirl_phase(axis, sense, odd[0], across, along, up)
                if weight > wp.float64(0.0):
                    mine = wp.float64(wp.abs(quanta[cell, across, along, up])) * swirl_share(axis, across, along, up)
                    share = wp.min(wp.int64(wp.floor(wp.float64(held) * mine / weight)), left)
                    quanta[cell, across, along, up] = sign * wp.int32(share)
                    left -= share
    for across in range(_BLOCK):
        for along in range(_BLOCK):
            for up in range(_BLOCK):
                if left > wp.int64(0):
                    quanta[cell, across, along, up] = quanta[cell, across, along, up] + sign
                    left -= wp.int64(1)


def start_swirls(state) -> None:
    """Make every living cell of the start a standing swirl, mirrored from neighbour to neighbour."""
    if state.step != 0 or state.count != state.side ** 3:
        raise ValueError("start_swirls winds a fresh start lattice, before any step or event")
    axis = min(int(draw_uniform_host(state.seed, 0, START_SWIRL_AXIS, 0) * _AXES), _AXES - 1)
    flip = 1 if draw_uniform_host(state.seed, 0, START_SWIRL_SENSE, 0) < 0.5 else -1
    wp.launch(_wind, dim=state.count, inputs=[state.kind, int(state.side), axis, flip],
              outputs=[state.swirl_axis, state.swirl_sense, state.quanta, state.phase], device=state.device)
