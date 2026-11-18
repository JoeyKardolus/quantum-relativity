"""Start energy: the start says energies and nothing else; every size follows from them.

Contract: `start_energy(state, settings)` fills `quanta` of every box of every cell of a
fresh grid in one kernel, then counts `cell_quanta` (count_cells.py).
  An energy volume's box holds `volume_quanta * exp(g) / mean(exp(g))` times
  `exp(inner_unevenness * normal)`, plus. `g` is the uneven field (uneven_field.py, from
  the grid's side and seed and the settings' `slope` and `unevenness`), one value per
  volume in rising id order; the mean is over all volumes, so before the inner factor
  the mean volume box holds `volume_quanta` (the inner factor's own mean is a little
  above 1: 1.005 at the default 0.1).
  A joint's box holds `joint_quanta * exp(joint_unevenness * cell normal)` times the
  same kind of inner factor, minus.
  Each is made whole (to the nearest whole number) and is at least 1: no box starts
  empty. An id that is no cell gets 0 in every box.
A normal is a standard normal by Box-Muller from two `draw_uniform` draws of step 0: the
box's normal from passes `START_VOLUME_A`, `START_VOLUME_B` and the box's index, a
joint's cell normal from `START_JOINT_A`, `START_JOINT_B` and the joint's id. So the
same seed gives the same start on any device.
Writes `quanta` and `cell_quanta`, nothing else: no phase, no size, no place.
Reads `kind`. Raises ValueError when the grid's volumes are not one per field value.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform
from physics.engine.axiom_1_energy.definitions.passes import (START_JOINT_A, START_JOINT_B, START_VOLUME_A,
                                                               START_VOLUME_B)
from physics.engine.axiom_1_energy.logic.count_cells import count_cells
from physics.engine.axiom_1_energy.logic.uneven_field import uneven_field
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, BOXES, ENERGY_VOLUME, NO_CELL

wp.set_module_options({"fuse_fp": False})

_NO_CELL = wp.constant(NO_CELL)
_ENERGY_VOLUME = wp.constant(ENERGY_VOLUME)
_BLOCK = wp.constant(BLOCK)
_BOXES = wp.constant(wp.uint64(BOXES))
_TURN = wp.constant(wp.float64(2.0 * math.pi))
_START_STEP = wp.constant(wp.uint32(0))
_VOLUME_A = wp.constant(wp.uint32(START_VOLUME_A))
_VOLUME_B = wp.constant(wp.uint32(START_VOLUME_B))
_JOINT_A = wp.constant(wp.uint32(START_JOINT_A))
_JOINT_B = wp.constant(wp.uint32(START_JOINT_B))
_WORD = 0xFFFFFFFF


@wp.func
def _normal(seed: wp.uint32, first_pass: wp.uint32, second_pass: wp.uint32, index: wp.uint64) -> wp.float64:
    """A standard normal for one thing, by Box-Muller from its two start draws."""
    first = draw_uniform(seed, _START_STEP, first_pass, index)
    second = draw_uniform(seed, _START_STEP, second_pass, index)
    return wp.sqrt(wp.float64(-2.0) * wp.log(wp.float64(1.0) - first)) * wp.cos(_TURN * second)


@wp.kernel
def _fill(kind: wp.array(dtype=wp.int32), volume_mean: wp.array(dtype=wp.float64), joint_quanta: wp.float64,
          joint_unevenness: wp.float64, inner_unevenness: wp.float64, seed: wp.uint32,
          quanta: wp.array4d(dtype=wp.int32)):
    cell, across, along, up = wp.tid()
    sign = kind[cell]
    if sign == _NO_CELL:
        quanta[cell, across, along, up] = 0
        return
    mean = volume_mean[cell]
    if sign != _ENERGY_VOLUME:
        mean = joint_quanta * wp.exp(joint_unevenness * _normal(seed, _JOINT_A, _JOINT_B, wp.uint64(cell)))
    box = wp.uint64(cell) * _BOXES + wp.uint64((across * _BLOCK + along) * _BLOCK + up)
    wanted = mean * wp.exp(inner_unevenness * _normal(seed, _VOLUME_A, _VOLUME_B, box))
    whole = wp.max(wp.int32(wp.floor(wanted + wp.float64(0.5))), 1)
    quanta[cell, across, along, up] = sign * whole


def _volume_means(state, settings) -> np.ndarray:
    """Per cell id: the mean quanta a box of that energy volume starts with (0 for every other id)."""
    volumes = np.flatnonzero(state.host("kind") == ENERGY_VOLUME)
    weight = np.exp(uneven_field(state.side, state.seed, settings.slope, settings.unevenness)).reshape(-1)
    if len(volumes) != weight.size:
        raise ValueError(f"the start has {len(volumes)} energy volumes but the field has {weight.size} values")
    means = np.zeros(state.capacity, dtype=np.float64)
    means[volumes] = settings.volume_quanta * weight / weight.mean()
    return means


def start_energy(state, settings) -> None:
    """Fill every box of the fresh grid with its whole signed quanta and count the cells."""
    means = wp.array(_volume_means(state, settings), dtype=wp.float64, device=state.device)
    wp.launch(_fill, dim=(state.count, BLOCK, BLOCK, BLOCK), device=state.device,
              inputs=[state.kind, means, wp.float64(settings.joint_quanta), wp.float64(settings.joint_unevenness),
                      wp.float64(settings.inner_unevenness), wp.uint32(state.seed & _WORD), state.quanta])
    count_cells(state)
