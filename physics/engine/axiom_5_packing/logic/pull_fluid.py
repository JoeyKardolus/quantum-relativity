"""Pull fluid: the pull moves plus energy by turning its fluid, across the whole cell.

Contract: `pull_fluid(state, settings)` is the one place plus energy feels the pull. For
every cell that feels it (sign_law.py: an energy volume) that holds energy:
  1. per direction, the mean pull of its neighbours there that hold energy (an id that
     is no cell and an empty cell are empty space and are not read). That mean is the
     pull at the cell's side in that direction, so two volumes that share a joint have
     the same pull at the side they share; the cell's own pull is the pull at its
     middle. `rise = mean - own`; 0 where it has no neighbour, and at its middle.
  2. every box turns its phase by `own * step_time + rise * tau`, `tau = step_time *
     rate[cell]` (the cell's local time), with `rise` taken between the middle and the
     sides the box stands between, in proportion (inside_pull.py). So the pull is one
     slope over the one fluid, inside a cell and from cell to cell: where the pull is
     deeper the fluid falls behind, and fluid flows from the wave that is ahead, so
     energy gathers speed toward the deeper pull, through its own cell and on through
     the joint (`cross`, axiom 4).
The cell's own pull turns its whole fluid because `cross` reads the phases of two cells
as they stand: a turn of the slope alone would make a step at every joint. That whole
turn is taken on counter time, the same for every cell, so a level added to the pull
everywhere turns every box of the universe alike and changes nothing (the level all
cells share sinks, `spread_pull`); the slope is on the cell's local time. Writes `phase`
of the cell's own boxes and `pull_felt[cell]` (the largest turn the slope gave any of
its boxes, a ledger). No quantum moves here. No thread reads a box of another cell.
"""
from __future__ import annotations

import warp as wp

from physics.engine.axiom_2_swirls.definitions.sign_law import feels_pull
from physics.engine.axiom_3_structure.definitions.direction_codes import CENTRE
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, NO_CELL
from physics.engine.axiom_3_structure.definitions.neighbour_of import neighbour_code, neighbour_count, neighbour_id
from physics.engine.axiom_5_packing.definitions.inside_pull import corner_code, corner_weight
from physics.engine.axiom_7_quantum.definitions.wrap_phase import wrap_phase

wp.set_module_options({"fuse_fp": False})

_NO_CELL = wp.constant(wp.int32(NO_CELL))
_CENTRE = wp.constant(wp.int32(CENTRE))
_CORNERS = 8
_DIRECTIONS = 27


@wp.kernel
def _side_rises(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
                adj_at: wp.array(dtype=wp.int32), adj_n: wp.array(dtype=wp.int32),
                adj_id: wp.array(dtype=wp.int32), adj_code: wp.array(dtype=wp.int32),
                pull: wp.array(dtype=wp.float64), rise: wp.array2d(dtype=wp.float64),
                number: wp.array2d(dtype=wp.int32)):
    """One thread per cell: how far the pull at each of its sides (its neighbours' mean there) lies above its own."""
    cell = wp.tid()
    if not feels_pull(kind[cell]) or cell_quanta[cell] <= wp.int64(0):
        return
    for index in range(neighbour_count(adj_n, cell)):
        other = neighbour_id(adj_at, adj_id, cell, index)
        if kind[other] == _NO_CELL or cell_quanta[other] <= wp.int64(0):
            continue
        code = neighbour_code(adj_at, adj_code, cell, index)
        rise[cell, code] = rise[cell, code] + pull[other]
        number[cell, code] = number[cell, code] + 1
    for code in range(_DIRECTIONS):
        if number[cell, code] > 0 and code != _CENTRE:
            rise[cell, code] = rise[cell, code] / wp.float64(number[cell, code]) - pull[cell]


@wp.kernel
def _pull_boxes(kind: wp.array(dtype=wp.int32), cell_quanta: wp.array(dtype=wp.int64),
                rise: wp.array2d(dtype=wp.float64), pull: wp.array(dtype=wp.float64),
                rate: wp.array(dtype=wp.float32), step_time: wp.float64,
                phase: wp.array4d(dtype=wp.float64), pull_felt: wp.array(dtype=wp.float32)):
    """One thread per box: turn its phase by the pull where it stands between the cell's middle and sides."""
    cell, across, along, up = wp.tid()
    if not feels_pull(kind[cell]) or cell_quanta[cell] <= wp.int64(0):
        return
    here = wp.float64(0.0)
    for corner in range(_CORNERS):
        here += corner_weight(across, along, up, corner) * rise[cell, corner_code(across, along, up, corner)]
    turn = here * step_time * wp.float64(rate[cell])
    phase[cell, across, along, up] = wrap_phase(phase[cell, across, along, up] + pull[cell] * step_time + turn)
    wp.atomic_max(pull_felt, cell, wp.float32(wp.abs(turn)))


def pull_fluid(state, settings) -> None:
    """Turn the fluid of every energy volume by the pull as it slopes across the cell."""
    rise = wp.zeros((state.count, _DIRECTIONS), dtype=wp.float64, device=state.device)
    number = wp.zeros((state.count, _DIRECTIONS), dtype=wp.int32, device=state.device)
    wp.launch(_side_rises, dim=state.count,
              inputs=[state.kind, state.cell_quanta, state.adj_at, state.adj_n, state.adj_id, state.adj_code,
                      state.pull],
              outputs=[rise, number], device=state.device)
    wp.launch(_pull_boxes, dim=(state.count, BLOCK, BLOCK, BLOCK),
              inputs=[state.kind, state.cell_quanta, rise, state.pull, state.rate, wp.float64(settings.step_time)],
              outputs=[state.phase, state.pull_felt], device=state.device)
