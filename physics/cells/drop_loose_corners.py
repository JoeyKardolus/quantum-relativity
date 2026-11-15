"""Drop loose corners: remove corners no tetrahedron uses and renumber the rest.

Contract: `drop_loose_corners(corners, tets)` returns (corners, renumber): only the
corners some tetrahedron uses, in their old order, and renumber (k,) with each old
corner's new index (-1 for a dropped one), to apply to tets and faces. Positions are
not changed.
"""
from __future__ import annotations

import numpy as np


def drop_loose_corners(corners: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corners some tetrahedron uses, and each old corner's new index."""
    used = np.zeros(len(corners), dtype=bool)
    used[tets.ravel()] = True
    renumber = np.where(used, np.cumsum(used) - 1, -1)
    return corners[used], renumber
