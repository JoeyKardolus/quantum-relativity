"""Extra centre: the mean place, along the axis between two lumps, of a lump's extra energy.

Contract: `extra_centre(extra, offsets)` takes two equal-length sequences: the extra
energy (quanta above an even volume's, never below 0) summed over each slice across the
axis, and each slice's place along the axis in volumes from the lump. Returns the
energy-weighted mean place as a float; 0.0 when there is no extra energy. Raises
ValueError for unequal lengths.
"""
from __future__ import annotations

import numpy as np


def extra_centre(extra, offsets) -> float:
    """Energy-weighted mean place of the extra energy along the axis (0.0 for none)."""
    extra, offsets = np.asarray(extra, dtype=np.float64), np.asarray(offsets, dtype=np.float64)
    if extra.shape != offsets.shape:
        raise ValueError(f"need equal lengths, got {extra.shape} and {offsets.shape}")
    total = float(extra.sum())
    return float((extra * offsets).sum() / total) if total > 0.0 else 0.0
