"""Blob centre: how much of the small blob's energy is still at its own volume, and where its extra energy's centre is.

Contract: `blob_centre(field, before, middle, small)` takes the volumes' quanta now and
what they would hold without the blob (two float arrays of one cube shape, volume_field:
the run and its twin at the same step), the big volume's index
along each axis (`middle`, the same on all three) and the small volume's index triple.
The blob is what each volume holds above `before`, never below 0, with the big volume
and the 26 volumes round it left out. Returns `(at_small, centre)`: the blob's quanta
at the small volume (float) and the blob's quanta-weighted mean place as three floats,
in volumes from the big one (wrapping: -half .. half). With no extra energy anywhere the
centre is (0, 0, 0).
"""
from __future__ import annotations

import numpy as np


def blob_centre(field: np.ndarray, before: np.ndarray, middle: int, small: tuple) -> tuple:
    """The extra quanta at the small blob's own volume, and the extra energy's centre from the big volume."""
    across = field.shape[0]
    extra = np.maximum(field - before, 0.0)
    offset = (np.indices(field.shape) - middle + across // 2) % across - across // 2
    extra[np.abs(offset).max(axis=0) <= 1] = 0.0
    weight = max(float(extra.sum()), 1.0)
    centre = [float((extra * offset[axis]).sum() / weight) for axis in range(3)]
    return float(extra[tuple(small)]), centre
