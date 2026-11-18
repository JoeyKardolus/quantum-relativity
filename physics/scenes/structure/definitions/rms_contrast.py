"""RMS contrast: how uneven a field of energy is, as one number.

Contract: `rms_contrast(field)` takes a float array of energy per place and returns the
root mean square of field / mean - 1 over all places, a float: 0 for an even field.
Raises ValueError for a field holding no energy (mean <= 0).
"""
from __future__ import annotations

import numpy as np


def rms_contrast(field: np.ndarray) -> float:
    """Root mean square of the field's contrast against its own mean."""
    field = np.asarray(field, dtype=np.float64)
    mean = float(field.mean())
    if mean <= 0.0:
        raise ValueError("the field holds no energy")
    return float(np.sqrt(np.mean((field / mean - 1.0) ** 2)))
