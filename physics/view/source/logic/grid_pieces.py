"""Grid pieces: the engine's grid now, as the blocks the viewer draws.

Contract: `grid_pieces(state, marks)` takes the engine's `Grid` and the start's packing
marks (`start_marks`) and returns a `Pieces` record (`cell_pieces`). It copies six
per-cell arrays to the host (kind, length, packing, torn_from, made_kind, swirl_axis)
and reads no box: Planck boxes are not space and are never drawn. It writes nothing.
"""
from __future__ import annotations

from physics.view.colours.logic.relative_packing import relative_packing
from physics.view.layout.definitions.pieces import Pieces
from physics.view.layout.logic.cell_pieces import cell_pieces
from physics.view.layout.logic.torn_lines import torn_lines


def grid_pieces(state, marks: dict) -> Pieces:
    """Every living cell of the engine's grid as its block in the warped grid."""
    kind = state.host("kind")
    lines, line_axis = torn_lines(state.side ** 3, kind, state.host("torn_from"), state.host("made_kind"),
                                  state.host("swirl_axis"))
    relative = relative_packing(kind, state.host("packing"), marks)
    return cell_pieces(state.side, kind, state.host("length"), relative, lines, line_axis)
