"""Grid triangles: the triangles to draw of a slice through the grid, or of the whole grid.

Contract: `grid_triangles(pieces, shown=SHOW_BOTH, layer=None)` takes a Pieces record
(`cell_pieces`) and returns (corners float32 (n, k, 3), colours uint8 (n, 4), pattern
int32 (m,), cells drawn): one row per drawn cell, its k corner points and its RGBA
(`cell_colours`: the hue is the kind, the opacity the packing), and the triangles of
every cell as corner numbers, three per triangle, the same for all cells. `shown` is
SHOW_ENERGY_VOLUMES, SHOW_JOINTS or SHOW_BOTH (any other value raises ValueError).
  layer k   the slice: every shown cell whose slot has lattice index k along z, as one
            flat four-cornered face through the middle of its block (k = 4, two
            triangles)
  None      the whole grid: every shown cell as its block (k = 8, six faces, twelve
            triangles). A drawer shades the faces by FACE_SHADE, by the axis they
            face, and raises the opacity to WHOLE_POWER (`cell_colours`), so thin
            cells are nearly clear and packed ones stand in them as solid masses.
"""
from __future__ import annotations

import numpy as np

from physics.view.cell_colours import cell_colours
from physics.view.cell_pieces import Pieces

SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOW_BOTH = 1, 2, 3
FACE_SHADE = (0.8, 1.0, 0.9)
_FACES = ((0, 2, 6, 4), (1, 3, 7, 5), (0, 1, 5, 4), (2, 3, 7, 6), (0, 1, 3, 2), (4, 5, 7, 6))
_BLOCK = np.array([face[corner] for face in _FACES for corner in (0, 1, 2, 0, 2, 3)], dtype=np.int32)
_FACE = np.array((0, 1, 2, 0, 2, 3), dtype=np.int32)


def grid_triangles(pieces: Pieces, shown: int = SHOW_BOTH, layer: int | None = None) -> tuple:
    """Corner points, colours and triangle pattern of the slice at `layer`, or of the whole grid."""
    if shown not in (SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOW_BOTH):
        raise ValueError(f"shown: {shown}, expected {SHOW_ENERGY_VOLUMES}, {SHOW_JOINTS} or {SHOW_BOTH}")
    keep = np.ones(len(pieces.cell), dtype=bool)
    if shown != SHOW_BOTH:
        keep &= pieces.kind == (1 if shown == SHOW_ENERGY_VOLUMES else -1)
    if layer is None:
        corners, pattern = pieces.corners[keep], _BLOCK
    else:
        keep &= pieces.place[:, 2] == layer
        blocks = pieces.corners[keep]
        corners, pattern = (0.5 * (blocks[:, :4] + blocks[:, 4:]))[:, (0, 1, 3, 2)], _FACE
    colours = cell_colours(pieces.kind[keep], pieces.packing[keep])
    return np.ascontiguousarray(corners, dtype=np.float32), colours, pattern, int(keep.sum())
