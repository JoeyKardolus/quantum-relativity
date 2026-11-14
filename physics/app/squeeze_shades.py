"""Per joint, how close it is to its limit (joint collapse), between 0 and 1.

Contract: `squeeze_shades(grid)` returns the grid's derived `squeeze` per
joint (0 at or above the joint's rest volume, 1 at the collapse limit;
physics.cells.step gives the formula), clipped to 0..1 and rounded to 3
decimals; a value that is not a number counts as 1.
"""
from __future__ import annotations

import numpy as np

from physics.cells import Grid


def squeeze_shades(grid: Grid) -> list[float]:
    """Each joint's nearness to joint collapse, 0..1."""
    squeeze = np.nan_to_num(np.asarray(grid.squeeze, dtype=float), nan=1.0)
    return np.round(np.clip(squeeze, 0.0, 1.0), 3).tolist()
