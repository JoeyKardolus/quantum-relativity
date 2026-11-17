"""Grid triangles: the triangles to draw of a slice through the grid, or of the whole grid.

Contract: `grid_triangles(pieces, shown=SHOW_BOTH, layer=None)` takes a Pieces record
(`cell_pieces`) and returns (positions float32 (n, 3), colours uint8 (n, 4), cells
drawn): three rows per triangle, RGB from `cell_colours`, alpha 255. `shown` is
SHOW_ENERGY_VOLUMES, SHOW_JOINTS or SHOW_BOTH (any other value raises ValueError).
  layer k   the slice: every shown cell whose slot has lattice index k along z, as one
            flat four-cornered face through the middle of its block (two triangles)
  None      the whole grid: every shown cell as its block, six faces shaded FACE_SHADE
            by the axis they face so the blocks read as solid. With both kinds shown
            only the slots on the outside of the opened grid are drawn: the grid has
            no gaps, so they hide everything behind them.
"""
from __future__ import annotations

import numpy as np

from physics.view.cell_colours import cell_colours
from physics.view.cell_pieces import Pieces

SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOW_BOTH = 1, 2, 3
FACE_SHADE = (0.8, 1.0, 0.9)
_FACES = ((0, 2, 6, 4), (1, 3, 7, 5), (0, 1, 5, 4), (2, 3, 7, 6), (0, 1, 3, 2), (4, 5, 7, 6))


def grid_triangles(pieces: Pieces, shown: int = SHOW_BOTH, layer: int | None = None) -> tuple:
    """Triangle corners and colours of the slice at `layer`, or of the whole grid."""
    if shown not in (SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOW_BOTH):
        raise ValueError(f"shown: {shown}, expected {SHOW_ENERGY_VOLUMES}, {SHOW_JOINTS} or {SHOW_BOTH}")
    keep = np.ones(len(pieces.cell), dtype=bool)
    if shown != SHOW_BOTH:
        keep &= pieces.kind == (1 if shown == SHOW_ENERGY_VOLUMES else -1)
    if layer is not None:
        keep &= pieces.place[:, 2] == layer
    elif shown == SHOW_BOTH:
        keep &= ((pieces.place == 0) | (pieces.place == pieces.side - 1)).any(axis=1)
    corners = pieces.corners[keep]
    colours = cell_colours(pieces.kind[keep], pieces.packing[keep])
    if layer is not None:
        faces = [(0.5 * (corners[:, :4] + corners[:, 4:]))[:, (0, 1, 3, 2)]]
        shades = [colours]
    else:
        faces = [corners[:, face] for face in _FACES]
        shades = [np.round(colours * FACE_SHADE[number // 2]).astype(np.uint8) for number in range(6)]
    quads, tints = np.concatenate(faces), np.concatenate(shades)
    positions = quads[:, (0, 1, 2, 0, 2, 3)].reshape(-1, 3).astype(np.float32)
    rgba = np.full((len(quads), 6, 4), 255, dtype=np.uint8)
    rgba[:, :, :3] = tints[:, None, :]
    return positions, rgba.reshape(-1, 4), int(keep.sum())
