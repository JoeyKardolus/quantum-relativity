"""Draw lines: thin straight lines blended into an image.

Contract: `draw_lines(picture, starts, ends, colours, opacity)` blends, into the uint8
(height, width, 3) `picture`, one pixel-wide line per row from `starts` to `ends` (float
(n, 2), x across, y down, in pixels) in that row's `colours` (n, 3) at `opacity` (0 to
1) over what is there. Parts outside the picture are left out. Where lines cross, the
later row is the one seen.
"""
from __future__ import annotations

import numpy as np


def draw_lines(picture: np.ndarray, starts: np.ndarray, ends: np.ndarray, colours: np.ndarray, opacity: float) -> None:
    """Blend one line per row into the picture."""
    if not len(starts) or opacity <= 0.0:
        return
    height, width = picture.shape[:2]
    steps = int(np.ceil(np.abs(ends - starts).max())) + 1
    tint = np.asarray(colours, dtype=np.float64) * opacity
    ground = picture.astype(np.float64) * (1.0 - opacity)
    for share in np.linspace(0.0, 1.0, steps + 1):
        point = np.floor(starts + (ends - starts) * share).astype(np.int64)
        inside = (point[:, 0] >= 0) & (point[:, 0] < width) & (point[:, 1] >= 0) & (point[:, 1] < height)
        across, down = point[inside, 0], point[inside, 1]
        picture[down, across] = np.round(ground[down, across] + tint[inside]).astype(np.uint8)
