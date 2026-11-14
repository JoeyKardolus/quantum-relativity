"""Seam joints: the three joints that meet at each seam.

Contract: `seam_joints(joints, seams, count)` returns (m, 3) int: per seam
(a, b, c), a < b < c, the rows of `joints` holding (a, b), (a, c) and (b, c),
in that order. Joint rows may be in either order (a, b) or (b, a); count is the
number of energy volumes. Raises ValueError when a seam's pair is not a joint
(the Grid keeps every seam's three pairs as joints).
"""
from __future__ import annotations

import numpy as np


def seam_joints(joints: np.ndarray, seams: np.ndarray, count: int) -> np.ndarray:
    """Per seam, the rows of its three joints (a, b), (a, c), (b, c)."""
    ends = np.sort(joints, axis=1)
    keys = ends[:, 0].astype(np.int64) * count + ends[:, 1]
    order = np.argsort(keys, kind="stable")
    wanted = np.column_stack([seams[:, 0] * count + seams[:, 1], seams[:, 0] * count + seams[:, 2],
                              seams[:, 1] * count + seams[:, 2]]).astype(np.int64)
    if wanted.size and not len(keys):
        raise ValueError("every pair of a seam must be a joint")
    slot = np.minimum(np.searchsorted(keys, wanted, sorter=order), max(len(order) - 1, 0))
    found = order[slot] if len(order) else np.zeros_like(slot)
    if (keys[found] != wanted).any():
        raise ValueError("every pair of a seam must be a joint")
    return found.reshape(-1, 3)
