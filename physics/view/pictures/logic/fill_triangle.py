"""Fill triangle: one triangle painted into an image.

Contract: `fill_triangle(picture, corners, colour)` sets, in the uint8 (height, width,
3) `picture`, every pixel whose centre lies in the triangle with float corners (3, 2)
(x across, y down, in pixels) to `colour`. Either winding; a triangle outside the
picture paints nothing.
"""
from __future__ import annotations

import numpy as np


def fill_triangle(picture: np.ndarray, corners: np.ndarray, colour: np.ndarray) -> None:
    """Paint every pixel whose centre lies in the triangle."""
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
