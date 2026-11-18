"""Start swirls: every cell of the start, energy volume and joint, begins as a swirl.

Contract: `start_swirls(state)` gives each living cell a line through the middle of its
block (`swirl_axis`: 0, 1 or 2) and a way round it (`swirl_sense`: +1 or -1), both drawn
per cell from the seed (`START_SWIRL_AXIS`, `START_SWIRL_SENSE`, the cell's id as index,
the state's step), so the grid has no chosen direction, and writes the swirl into the
cell's fluid: every box gets the phase `swirl_phase` gives it, one whole turn round the
axis. From then on nothing turns the swirl: the fluid goes round by its own phase
(`flow_inside`, axiom 1), on the cell's local time. An id that is no cell is left alone.
Reads: `kind`, `seed`, `step`. Writes: `swirl_axis`, `swirl_sense`, `phase`. No quantum
moves here.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform
from physics.engine.axiom_1_energy.definitions.passes import START_SWIRL_AXIS, START_SWIRL_SENSE
from physics.engine.axiom_2_swirls.definitions.swirl_phase import swirl_phase
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, NO_CELL

wp.set_module_options({"fuse_fp": False})

_NO_CELL = wp.constant(wp.int32(NO_CELL))
_AXIS_PASS = wp.constant(wp.uint32(START_SWIRL_AXIS))
_SENSE_PASS = wp.constant(wp.uint32(START_SWIRL_SENSE))
_AXES = wp.constant(wp.float64(3.0))
_HALF = wp.constant(wp.float64(0.5))


@wp.kernel
def _draw_swirls(kind: wp.array(dtype=wp.int32), seed: wp.uint32, step: wp.uint32,
                 swirl_axis: wp.array(dtype=wp.int32), swirl_sense: wp.array(dtype=wp.int32)):
    """Draw one cell's axis (each of the three as likely) and its sense (each as likely)."""
    cell = wp.tid()
    if kind[cell] == _NO_CELL:
        return
    axis = wp.int32(wp.floor(draw_uniform(seed, step, _AXIS_PASS, wp.uint64(cell)) * _AXES))
    swirl_axis[cell] = wp.min(axis, 2)
    if draw_uniform(seed, step, _SENSE_PASS, wp.uint64(cell)) < _HALF:
        swirl_sense[cell] = 1
    else:
        swirl_sense[cell] = -1


@wp.kernel
def _wind_fluid(kind: wp.array(dtype=wp.int32), swirl_axis: wp.array(dtype=wp.int32),
                swirl_sense: wp.array(dtype=wp.int32), phase: wp.array4d(dtype=wp.float64)):
    """One thread per box: the phase that takes the cell's fluid once round its axis."""
    cell, across, along, up = wp.tid()
    if kind[cell] == _NO_CELL:
        return
    phase[cell, across, along, up] = swirl_phase(swirl_axis[cell], swirl_sense[cell], across, along, up)


def start_swirls(state) -> None:
    """Give every living cell a swirl axis and sense from the seed and wind its fluid once round the axis."""
    wp.launch(_draw_swirls, dim=state.count,
              inputs=[state.kind, wp.uint32(state.seed), wp.uint32(state.step)],
              outputs=[state.swirl_axis, state.swirl_sense], device=state.device)
    wp.launch(_wind_fluid, dim=(state.count, BLOCK, BLOCK, BLOCK),
              inputs=[state.kind, state.swirl_axis, state.swirl_sense], outputs=[state.phase],
              device=state.device)
