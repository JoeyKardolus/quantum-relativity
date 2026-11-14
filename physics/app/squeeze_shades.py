"""Per joint, how close its push is to the push limit, between 0 and 1.

Contract: `squeeze_shades(grid, settings)` returns one value per joint: the
push its squeeze asks for (joint_push * squeeze) as a share of push_limit,
clipped to 0..1 and rounded to 3 decimals. 1 means the joint is at the limit:
asked for more, it collapses (the rule in physics.cells.step). With a push
limit of 0 every squeezed joint is at the limit.
"""
from __future__ import annotations

import numpy as np

from physics.cells import CellSettings, Grid


def squeeze_shades(grid: Grid, settings: CellSettings) -> list[float]:
    """Each joint's asked-for push as a share of the push limit."""
    asked = settings.joint_push * np.asarray(grid.squeeze, dtype=float)
    if settings.push_limit > 0:
        share = asked / settings.push_limit
    else:
        share = (asked > 0).astype(float)
    return np.round(np.clip(np.nan_to_num(share, nan=1.0), 0.0, 1.0), 3).tolist()
