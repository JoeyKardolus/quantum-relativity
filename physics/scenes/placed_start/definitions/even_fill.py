"""Even fill: a whole number of quanta spread as evenly as whole numbers allow over a block.

Contract: `even_fill(total)` returns an int32 array of shape (BLOCK, BLOCK, BLOCK)
(physics.engine BLOCK, 6) whose boxes sum to exactly `total`: every box holds
total // 216 and the first total % 216 boxes (in raster order) one more. Raises
ValueError for a total below 0.
"""
from __future__ import annotations

import numpy as np

from physics.engine import BLOCK, BOXES


def even_fill(total: int) -> np.ndarray:
    """A block of whole quanta summing to `total`, no two boxes more than one apart."""
    total = int(total)
    if total < 0:
        raise ValueError(f"a volume holds 0 or more quanta, got {total}")
    boxes = np.full(BOXES, total // BOXES, dtype=np.int32)
    boxes[:total % BOXES] += 1
    return boxes.reshape(BLOCK, BLOCK, BLOCK)
