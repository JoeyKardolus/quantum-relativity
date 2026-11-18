"""Slice picture: a slice through the grid as an image, without a window.

Contract: `slice_picture(pieces, layer, low, high, width=WIDTH, shown=SHOW_BOTH)` draws
the slice at lattice index `layer` along z (`grid_triangles`) looking straight down z,
with no perspective, and returns uint8 (height, width, 3) RGB on black, row 0 at the
top. The picture shows the rectangle from `low` to `high` (x and y, grid lengths; pass
the same two for every frame of a run and nothing jumps between frames); height
follows from its shape. Every cell is filled, hard-edged, with its one colour as seen
over black (its hue times its opacity): a thin cell fades into the ground, a packed one
stands out solid. A pixel takes the cell its centre lies in.
"""
from __future__ import annotations

import numpy as np

from physics.view.drawing.definitions.shown_kinds import SHOW_BOTH
from physics.view.drawing.logic.grid_triangles import grid_triangles
from physics.view.layout.definitions.pieces import Pieces
from physics.view.pictures.definitions.picture_marks import WIDTH
from physics.view.pictures.logic.fill_triangle import fill_triangle


def slice_picture(pieces: Pieces, layer: int, low, high, width: int = WIDTH, shown: int = SHOW_BOTH) -> np.ndarray:
    """An image of one slice of the grid."""
    low, high = np.asarray(low, dtype=np.float64)[:2], np.asarray(high, dtype=np.float64)[:2]
    scale = width / (high[0] - low[0])
    height = max(int(round((high[1] - low[1]) * scale)), 1)
    corners, colours, pattern, _ = grid_triangles(pieces, shown, layer)
    picture = np.zeros((height, width, 3), dtype=np.uint8)
    points = (corners[:, :, :2].astype(np.float64) - low) * scale
    points[:, :, 1] = height - points[:, :, 1]
    over_black = np.round(colours[:, :3] * (colours[:, 3:] / 255.0)).astype(np.uint8)
    for cell_points, colour in zip(points, over_black):
        for triangle in pattern.reshape(-1, 3):
            fill_triangle(picture, cell_points[triangle], colour)
    return picture
