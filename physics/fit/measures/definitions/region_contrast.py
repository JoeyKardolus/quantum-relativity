"""Region contrast: how uneven the energy is over regions larger than one cell.

Contract: `region_contrast(field)` takes the energy volumes' quanta as a cube (one value
per volume place of the start lattice, 0 for a volume that is gone) and returns the rms
of `block / mean block - 1` over blocks of 2 x 2 x 2 volumes. 0 when every block holds
the same. 0 too when the cube's side is odd or under 4 (no blocks to compare) or when
it holds nothing.
"""
from __future__ import annotations

import numpy as np


def region_contrast(field: np.ndarray) -> float:
    """rms of (block amount / mean - 1) over blocks of 2 x 2 x 2 volumes."""
    side = field.shape[0]
    if side % 2 or side < 4:
        return 0.0
    blocks = side // 2
    amount = field.reshape(blocks, 2, blocks, 2, blocks, 2).sum(axis=(1, 3, 5)).astype(np.float64)
    if amount.mean() <= 0.0:
        return 0.0
    return float(np.sqrt(np.mean((amount / amount.mean() - 1.0) ** 2)))
