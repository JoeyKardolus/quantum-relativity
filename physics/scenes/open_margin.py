"""Open margin: the empty margin an open-space scene starts with, as wide as the filled cube.

Contract: `open_margin(cells_per_side, boxes_per_cell_side)` returns the width in Planck
boxes of the filled cube physics.engine.new_universe builds when it does not wrap,
((cells_per_side - 1) // 2) * (boxes_per_cell_side + 1) + 1, to pass as its `margin`
(bible "How results are trusted": the sim starts as a filled cube inside an empty margin;
physics/CLAUDE.md: the margin is as wide as the filled cube). new_universe's own default
is margin 0, a universe that wraps round or has walls, where space cannot grow outward.
Raises ValueError for fewer than 3 cells per side or 2 boxes per cell side.
"""
from __future__ import annotations


def open_margin(cells_per_side: int, boxes_per_cell_side: int) -> int:
    """The filled cube's width in boxes, the margin an open-space start gets on every side."""
    if cells_per_side < 3 or boxes_per_cell_side < 2:
        raise ValueError(f"need at least 3 cells per side of at least 2 boxes, got {cells_per_side}, "
                         f"{boxes_per_cell_side}")
    return ((cells_per_side - 1) // 2) * (boxes_per_cell_side + 1) + 1
