"""Draw whole: the direct pull, whole plus quanta drawn out of cells and into others at once.

Contract: `draw_whole(state, take, give, heat_change)` takes
one number per id in use (host arrays) and writes `quanta` and `heat_quanta`
of the energy volumes named, in two launches:
  take  `take[cell]` whole plus quanta leave the cell's boxes in proportion to what each
        holds: each box gives the whole part of its share, and the quanta still owed
        leave one a box, from a box drawn for the cell (`SETTLE_DEAL`) onward, from boxes
        that still hold some. Never more than the cell holds. The boxes keep their phases.
  give  `give[cell]` whole plus quanta arrive, dealt the same way over the boxes in
        proportion to what each holds after the taking (evenly into an empty cell).
        They come to rest in the fluid they join: every box keeps its phase, so the
        taking cell's own swirl stands and no motion is made.
  heat  `heat_change[cell]` is added to `heat_quanta`: heat moves with the quanta.
At once and at any distance: this is the pull between a split's halves, and nothing
else calls it. The caller sees to it that `take` sums to `give`, so no total changes.
`cell_quanta` is not written (`count_cells` does). A cell that is no energy volume is
left alone.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.axiom_1_energy.definitions.cell_sum import cell_sum
from physics.engine.axiom_1_energy.definitions.draw_uniform import draw_uniform
from physics.engine.axiom_1_energy.definitions.passes import SETTLE_DEAL
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, BOXES, ENERGY_VOLUME

wp.set_module_options({"fuse_fp": False})

_BLOCK = wp.constant(BLOCK)
_BOXES = wp.constant(BOXES)
_ENERGY_VOLUME = wp.constant(wp.int32(ENERGY_VOLUME))
_SETTLE_DEAL = wp.constant(wp.uint32(SETTLE_DEAL))
_MANY = wp.constant(wp.float64(BOXES))
_EVERY = wp.constant(wp.int64(BOXES))
_SIDES = 2
_WORD = 0xFFFFFFFF


@wp.func
def _box(number: wp.int32) -> wp.vec3i:
    """The box with this number in a block."""
    return wp.vec3i(number / (_BLOCK * _BLOCK), (number / _BLOCK) % _BLOCK, number % _BLOCK)


@wp.func
def _first_box(seed: wp.uint32, step: wp.uint32, cell: wp.int32, side: wp.int32) -> wp.int32:
    """The box that gives (side 0) or gets (side 1) the first of the quanta left over."""
    drawn = draw_uniform(seed, step, _SETTLE_DEAL, wp.uint64(cell) * wp.uint64(_SIDES) + wp.uint64(side))
    return wp.min(wp.int32(wp.floor(drawn * _MANY)), _BOXES - 1)


@wp.kernel
def _take(kind: wp.array(dtype=wp.int32), take: wp.array(dtype=wp.int64), seed: wp.uint32, step: wp.uint32,
          quanta: wp.array4d(dtype=wp.int32)):
    cell = wp.tid()
    if kind[cell] != _ENERGY_VOLUME or take[cell] <= wp.int64(0):
        return
    held = cell_sum(quanta, cell)
    if held <= wp.int64(0):
        return
    wanted = wp.min(take[cell], held)
    owed = wanted
    for number in range(_BOXES):
        box = _box(number)
        has = wp.int64(quanta[cell, box[0], box[1], box[2]])
        share = wanted * has / held
        quanta[cell, box[0], box[1], box[2]] = wp.int32(has - share)
        owed -= share
    first = _first_box(seed, step, cell, 0)
    for offset in range(_BOXES):
        box = _box((first + offset) % _BOXES)
        if owed > wp.int64(0) and quanta[cell, box[0], box[1], box[2]] > 0:
            quanta[cell, box[0], box[1], box[2]] -= 1
            owed -= wp.int64(1)


@wp.kernel
def _give(kind: wp.array(dtype=wp.int32), give: wp.array(dtype=wp.int64), heat_change: wp.array(dtype=wp.int64),
          seed: wp.uint32, step: wp.uint32, quanta: wp.array4d(dtype=wp.int32),
          heat_quanta: wp.array(dtype=wp.int64)):
    cell = wp.tid()
    heat_quanta[cell] += heat_change[cell]
    arriving = give[cell]
    if kind[cell] != _ENERGY_VOLUME or arriving <= wp.int64(0):
        return
    held = cell_sum(quanta, cell)
    first = _first_box(seed, step, cell, 1)
    owed = arriving
    for number in range(_BOXES):
        box = _box(number)
        has = wp.int64(quanta[cell, box[0], box[1], box[2]])
        if held > wp.int64(0):
            owed -= arriving * has / held
        else:
            owed -= arriving / _EVERY
    for offset in range(_BOXES):
        box = _box((first + offset) % _BOXES)
        has = wp.int64(quanta[cell, box[0], box[1], box[2]])
        share = arriving / _EVERY
        if held > wp.int64(0):
            share = arriving * has / held
        if owed > wp.int64(0) and (has > wp.int64(0) or held == wp.int64(0)):
            share += wp.int64(1)
            owed -= wp.int64(1)
        if share > wp.int64(0):
            quanta[cell, box[0], box[1], box[2]] = wp.int32(has + share)


def _sent(state, values, dtype):
    """One host number per id in use, on the device."""
    return wp.array(np.asarray(values), dtype=dtype, device=state.device)


def draw_whole(state, take, give, heat_change) -> None:
    """Take whole plus quanta out of the cells named and give whole plus quanta, at rest, to others."""
    cells, seed, step = int(state.count), wp.uint32(state.seed & _WORD), wp.uint32(state.step & _WORD)
    wp.launch(_take, dim=cells, device=state.device,
              inputs=[state.kind, _sent(state, take, wp.int64), seed, step, state.quanta])
    wp.launch(_give, dim=cells, device=state.device,
              inputs=[state.kind, _sent(state, give, wp.int64), _sent(state, heat_change, wp.int64), seed, step,
                      state.quanta, state.heat_quanta])
