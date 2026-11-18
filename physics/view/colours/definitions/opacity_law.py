"""Opacity law: how solid a cell is drawn, from how packed it is.

Contract: `opacity_law(relative)` takes packings as multiples of the start's mean
(`relative_packing`) and returns float64 opacities in 0 to 1: relative / SOLID, cut off
at 1. A straight line through 0, so an emptied cell is fully clear, the start's mean is
0.25 and a cell SOLID times as packed, or more, is solid. A packing that is not a
number counts as 0. The scale is fixed: an opacity means the same packing in every
frame.
"""
from __future__ import annotations

import numpy as np

from physics.view.colours.definitions.palette import SOLID


def opacity_law(relative: np.ndarray) -> np.ndarray:
    """Opacity 0 to 1 of every cell from its packing over the start's mean."""
    return np.clip(np.nan_to_num(np.asarray(relative, dtype=np.float64), nan=0.0) / SOLID, 0.0, 1.0)
