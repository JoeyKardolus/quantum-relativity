"""Grid pieces: the engine's grid now, as the blocks the viewer draws.

Contract: `grid_pieces(state, marks, layer=None)` takes the engine's `Grid` and the
start's packing marks (`start_marks`) and returns a `Pieces` record (`cell_pieces`)
whose packing is each cell's over its kind's mark (`relative_packing`): the living
cells of lattice layer `layer`, or with None only the cells packed enough to show in
the whole grid (`packed_cells`). It copies six per-cell arrays to the host (kind,
length, packing, torn_from, made_kind, swirl_axis) and reads no box: Planck boxes are
not space and are never drawn. It writes nothing.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.view.colours.logic.packed_cells import packed_cells
from physics.view.colours.logic.relative_packing import relative_packing
from physics.view.layout.definitions.pieces import Pieces
from physics.view.layout.logic.cell_pieces import cell_pieces
from physics.view.layout.logic.torn_lines import torn_lines


def grid_pieces(state, marks: dict, layer: int | None = None) -> Pieces:
    """One layer of the engine's grid, or its packed cells, as blocks of the drawn grid."""
    kind, packing = state.host("kind"), state.host("packing")
    lines, line_axis = torn_lines(state.side ** 3, kind, state.host("torn_from"), state.host("made_kind"),
                                  state.host("swirl_axis"))
    only = packed_cells(kind, packing, marks) if layer is None else None
    pieces = cell_pieces(state.side, kind, state.host("length"), packing, lines, line_axis, layer, only)
    return replace(pieces, packing=relative_packing(pieces.kind, pieces.packing, marks).astype(np.float32))
