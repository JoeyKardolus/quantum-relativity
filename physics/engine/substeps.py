"""Substeps: how many equal fluid substeps each cell takes this step, set by its own steepest face.

Contract: `substeps(state, settings)` reads, per cell, the steepest face: the largest
|turn rate difference| between two face-neighbouring boxes of that cell (the hops act
only there, hop.py), a box's turn rate being fluid_potential * its cell's clock rate
(on_site.py). Over one substep of length step_time / n that face's phase difference
turns by steepest * step_time / n; n is the smallest whole number keeping it at or below
settings.max_turn, at least 1 and at most settings.max_substeps. Numerics, not a rule
(owner 2026-11-16, decision E): the steep push turned a squeezed joint's phase by
radians a step, the split fluid step then scrambled the phases, and collapse ran away;
cutting the step where it is steep integrates the push stably at any squeeze, and the
step's counter time stays step_time. The fluid never moves from one cell into another
(hop.py), so each cell takes its own count (2026-11-17; before, a whole universe took
its steepest cell's count, and at 32 places a side that was 4 to 13 substeps for
everyone while 99 in 100 cells needed 1). Writes state.substeps, a Warp int32 array
(universes, capacity), the plan per cell: its size is the cell's count; it is negative
when the cell's substeps after the first are run over a list of its boxes
(later_substeps.py), which is every cell needing more than one, unless those cells hold
more than a quarter of all boxes (then the launches over all boxes run every substep).
Sets state.later: a dict with "dense" (how many launches over all boxes this step
needs: 1 with a list), "boxes" (None, or a Warp int32 array: the listed cells' boxes as
indices into the stacked arrays, cells of the highest count first) and "ends" (per
substep index from 1 the number of listed boxes whose cell still steps then). The list
is filled from state.cell_boxes (cell_count.py, counted this step). Adds 1 to state.capped for each universe with a cell that
needed more than max_substeps, and returns the largest count per universe (int32
numpy). settings.max_turn = 0 gives 1 everywhere (the coarse step) without reading the
boxes.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.face_step import face_step
from physics.engine.fluid_potential import fluid_potential
from physics.engine.reach import reach
from physics.engine.settings import Settings
from physics.engine.state import State

DENSE_SHARE = 0.25


@wp.kernel
def _steepest(quanta: wp.array3d(dtype=wp.int32), owner: wp.array3d(dtype=wp.int32), pull: wp.array3d(dtype=wp.float32),
              rates: wp.array2d(dtype=wp.float32), pushes: wp.array2d(dtype=wp.float32),
              packings: wp.array2d(dtype=wp.float32), pressure: float, side: int, wrap: int,
              steepest: wp.array2d(dtype=wp.float32)):
    ti, tj, tk = wp.tid()
    held = quanta[ti, tj, tk]
    if held == 0:
        return
    home = ti // side
    here = wp.vec3i(ti, tj, tk)
    cell = owner[ti, tj, tk]
    own = fluid_potential(held, pull[ti, tj, tk], pushes[home, cell], packings[home, cell], pressure) * rates[home, cell]
    largest = float(0.0)
    for face in range(3):
        there, valid = reach(here, face_step(2 * face), home, side, wrap)
        if valid and owner[there[0], there[1], there[2]] == cell:
            other = fluid_potential(quanta[there[0], there[1], there[2]], pull[there[0], there[1], there[2]],
                                    pushes[home, cell], packings[home, cell], pressure) * rates[home, cell]
            largest = wp.max(largest, wp.abs(other - own))
    if largest > 0.0:
        wp.atomic_max(steepest, home, cell, largest)


@wp.kernel
def _plan(steepest: wp.array2d(dtype=wp.float32), step_time: wp.float64, max_turn: wp.float64, most: int,
          plans: wp.array2d(dtype=wp.int32), needed: wp.array(dtype=wp.int32)):
    home, cell = wp.tid()
    need = wp.max(1, int(wp.ceil(wp.float64(steepest[home, cell]) * step_time / max_turn)))
    plans[home, cell] = wp.min(need, most)
    if need > 1:
        wp.atomic_max(needed, home, need)


@wp.kernel
def _list(owner: wp.array3d(dtype=wp.int32), starts: wp.array2d(dtype=wp.int32),
          taken: wp.array2d(dtype=wp.int32), side: int, boxes: wp.array(dtype=wp.int32)):
    ti, tj, tk = wp.tid()
    cell = owner[ti, tj, tk]
    if cell < 0:
        return
    home = ti // side
    start = starts[home, cell]
    if start < 0:
        return
    place = start + wp.atomic_add(taken, home, cell, 1)
    if place < boxes.shape[0]:
        boxes[place] = (ti * side + tj) * side + tk


@wp.kernel
def _listed(homes: wp.array(dtype=wp.int32), cells: wp.array(dtype=wp.int32), plans: wp.array2d(dtype=wp.int32)):
    item = wp.tid()
    plans[homes[item], cells[item]] = -plans[homes[item], cells[item]]


def substeps(state: State, settings: Settings) -> np.ndarray:
    """Each cell's substep count from its steepest face, written to state.substeps; the listed boxes in state.later."""
    device = state.quanta.device
    shape = state.kind.shape
    state.later = {"dense": 1, "boxes": None, "ends": np.zeros(1, dtype=np.int64)}
    if settings.max_turn <= 0.0:
        state.substeps = wp.ones(shape, dtype=wp.int32, device=device)
        return np.ones(state.universes, dtype=np.int32)
    steepest = wp.zeros(shape, dtype=wp.float32, device=device)
    wp.launch(_steepest, dim=state.shape, device=device,
              inputs=[state.quanta, state.owner, state.pull, state.cells_d["rate"], state.cells_d["push"],
                      state.cells_d["packing"], float(settings.push), state.side, int(state.wrap), steepest])
    plans = wp.empty(shape, dtype=wp.int32, device=device)
    needed = wp.ones(state.universes, dtype=wp.int32, device=device)
    wp.launch(_plan, dim=shape, device=device,
              inputs=[steepest, wp.float64(settings.step_time), wp.float64(settings.max_turn),
                      int(settings.max_substeps), plans, needed])
    state.substeps = plans
    most = needed.numpy()
    state.capped += most > settings.max_substeps
    counts = np.minimum(most, settings.max_substeps).astype(np.int32)
    if counts.max() > 1:
        _list_boxes(state, plans)
    return counts


def _list_boxes(state: State, plans) -> None:
    """List the boxes of every cell that needs more than one substep, highest count first; too many: all launches dense."""
    device = state.quanta.device
    held = plans.numpy()
    homes, cells = np.nonzero(held > 1)
    order = np.argsort(-held[homes, cells], kind="stable")
    homes, cells = homes[order], cells[order]
    counts = held[homes, cells]
    sizes = state.cell_boxes[homes, cells].astype(np.int64)
    total = int(sizes.sum())
    if total == 0 or total > DENSE_SHARE * state.universes * state.side ** 3:
        state.later["dense"] = int(counts.max())
        return
    starts = np.full(held.shape, -1, dtype=np.int32)
    starts[homes, cells] = np.cumsum(sizes) - sizes
    boxes = wp.empty(total, dtype=wp.int32, device=device)
    wp.launch(_list, dim=state.shape, device=device,
              inputs=[state.owner, wp.array(starts, dtype=wp.int32, device=device),
                      wp.zeros(held.shape, dtype=wp.int32, device=device), state.side, boxes])
    wp.launch(_listed, dim=homes.size, device=device,
              inputs=[wp.array(homes.astype(np.int32), dtype=wp.int32, device=device),
                      wp.array(cells.astype(np.int32), dtype=wp.int32, device=device), plans])
    ends = np.cumsum(sizes)
    state.later["boxes"] = boxes
    state.later["ends"] = np.array([0] + [int(ends[np.searchsorted(-counts, -sub_index, side="left") - 1])
                                          for sub_index in range(1, int(counts.max()))], dtype=np.int64)
