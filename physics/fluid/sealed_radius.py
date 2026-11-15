"""Sealed radius: the size of the largest sealed region, a black hole's horizon, measured in the structure.

Contract: `sealed_radius(compression, length, seal)` takes numpy arrays per box of B
compression (b / b_rest) and local length (local_length.py) and the seal compression;
it returns the radius (3 V / (4 pi))**(1/3) of the sphere whose volume V is the
summed length cubed of the boxes in the largest 6-connected region with compression at
or past the seal (nothing leaks out of it: bible milestone 5), and 0.0 when no box is
sealed. Largest by that volume. Measured from lengths, not from box counts, so a
sealed region that packs more energy into the same boxes measures smaller.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def sealed_radius(compression: np.ndarray, length: np.ndarray, seal: float) -> float:
    """Equivalent-sphere radius of the largest sealed region's measured volume."""
    labels, count = ndimage.label(compression >= seal)
    if count == 0:
        return 0.0
    volumes = ndimage.sum(np.where(labels > 0, length, 0.0) ** 3, labels, index=np.arange(1, count + 1))
    return float((3.0 * float(np.max(volumes)) / (4.0 * math.pi)) ** (1.0 / 3.0))
