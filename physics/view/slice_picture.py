"""Slice picture: a slice through the grid as an image, without a window.

Contract: `slice_picture(pieces, layer, low, high, width=1100, shown=SHOW_BOTH)` draws
the slice at lattice index `layer` along z (`grid_triangles`) looking straight down z,
with no perspective, and returns uint8 (height, width, 3) RGB on black, row 0 at the
top. The picture shows the rectangle from `low` to `high` (x and y, grid lengths; pass
the same two for every frame of a run and nothing jumps between frames); height
follows from its shape. Every cell is filled with its one flat colour, hard-edged, a
pixel taking the cell its centre lies in.
"""
from __future__ import annotations

import numpy as np

from physics.view.cell_pieces import Pieces
from physics.view.grid_triangles import SHOW_BOTH, grid_triangles


def slice_picture(pieces: Pieces, layer: int, low, high, width: int = 1100, shown: int = SHOW_BOTH) -> np.ndarray:
    """An image of one slice of the grid."""
    low, high = np.asarray(low, dtype=np.float64)[:2], np.asarray(high, dtype=np.float64)[:2]
    scale = width / (high[0] - low[0])
    height = max(int(round((high[1] - low[1]) * scale)), 1)
    positions, colours, _ = grid_triangles(pieces, shown, layer)
    picture = np.zeros((height, width, 3), dtype=np.uint8)
    points = (positions[:, :2].astype(np.float64) - low) * scale
    points[:, 1] = height - points[:, 1]
    for first in range(0, len(points), 3):
        _fill(picture, points[first:first + 3], colours[first, :3])
    return picture


def _fill(picture: np.ndarray, corners: np.ndarray, colour: np.ndarray) -> None:
    """Fill one triangle: every pixel whose centre lies in it."""
    height, width = picture.shape[:2]
    left, top = np.floor(corners.min(axis=0)).astype(int)
    right, bottom = np.ceil(corners.max(axis=0)).astype(int)
    left, right, top, bottom = max(left, 0), min(right, width), max(top, 0), min(bottom, height)
    if left >= right or top >= bottom:
        return
    across, down = np.meshgrid(np.arange(left, right) + 0.5, np.arange(top, bottom) + 0.5)
    inside_plus = np.ones(across.shape, dtype=bool)
    inside_minus = np.ones(across.shape, dtype=bool)
    for one in range(3):
        start, end = corners[one], corners[(one + 1) % 3]
        turn = (end[0] - start[0]) * (down - start[1]) - (end[1] - start[1]) * (across - start[0])
        inside_plus &= turn >= 0.0
        inside_minus &= turn <= 0.0
    picture[top:bottom, left:right][inside_plus | inside_minus] = colour
