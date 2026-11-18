"""Grid triangles: the triangles to draw of the whole grid, or of a slice through it.

Contract: `grid_triangles(pieces, shown=SHOW_BOTH, layer=None)` takes a `Pieces` record
and returns (corners float32 (n, k, 3), colours uint8 (n, 4), pattern int32 (m,), cells
drawn): one row per drawn cell, its k corner points and its RGBA (`cell_colours`), and
the triangles of every cell as corner numbers, the same for all cells. `shown` is one
of SHOWN (any other value raises ValueError).
  layer None  the whole grid: every shown cell as its block (k = 8, BLOCK_PATTERN). A
              drawer shades the faces by FACE_SHADE and raises the opacity to
              WHOLE_POWER, so thin cells are nearly clear and packed ones stand in
              them as solid masses.
  layer k     the slice: every shown cell whose slot has lattice index k along z, as
              one flat four-cornered face through the middle of its block (k = 4,
              FACE_PATTERN)
"""
from __future__ import annotations

import numpy as np

from physics.view.colours.logic.cell_colours import cell_colours
from physics.view.drawing.definitions.block_faces import BLOCK_PATTERN, FACE_PATTERN
from physics.view.drawing.definitions.shown_kinds import SHOW_BOTH, SHOW_ENERGY_VOLUMES, SHOWN
from physics.view.layout.definitions.kinds import ENERGY_VOLUME, JOINT
from physics.view.layout.definitions.pieces import Pieces


def grid_triangles(pieces: Pieces, shown: int = SHOW_BOTH, layer: int | None = None) -> tuple:
    """Corner points, colours and triangle pattern of the whole grid, or of the slice at `layer`."""
    if shown not in SHOWN:
        raise ValueError(f"shown: {shown}, expected one of {SHOWN}")
    keep = np.ones(len(pieces.cell), dtype=bool)
    if shown != SHOW_BOTH:
        keep &= pieces.kind == (ENERGY_VOLUME if shown == SHOW_ENERGY_VOLUMES else JOINT)
    if layer is None:
        corners, pattern = pieces.corners[keep], BLOCK_PATTERN
    else:
        keep &= pieces.place[:, 2] == layer
        blocks = pieces.corners[keep]
        corners, pattern = (0.5 * (blocks[:, :4] + blocks[:, 4:]))[:, (0, 1, 3, 2)], FACE_PATTERN
    colours = cell_colours(pieces.kind[keep], pieces.packing[keep])
    return np.ascontiguousarray(corners, dtype=np.float32), colours, pattern, int(keep.sum())
