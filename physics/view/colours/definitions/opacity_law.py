"""Opacity law: how solid a cell's fill is drawn, from how packed it is.

Contract: `opacity_law(relative)` takes packings as multiples of the start's mean
(`relative_packing`) and returns float64 opacities in 0 to 1: (relative - 1) /
(SOLID - 1), cut off at 0 and 1. So the calm background, at the start's mean or
thinner, is fully clear, and a cell turns solid in a straight line up to SOLID times
the mean. A packing that is not a number counts as clear. The scale is fixed: an
opacity means the same packing in every frame.
"""
from __future__ import annotations

import numpy as np

from physics.view.colours.definitions.palette import SOLID


def opacity_law(relative: np.ndarray) -> np.ndarray:
    """Opacity 0 to 1 of every cell's fill from its packing over the start's mean."""
    return np.clip((np.nan_to_num(np.asarray(relative, dtype=np.float64), nan=0.0) - 1.0) / (SOLID - 1.0), 0.0, 1.0)
