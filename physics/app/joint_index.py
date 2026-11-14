"""Which joint, if any, lies between each given pair of energy volumes.

Contract: `joint_index(joints, count, pairs)` takes the (k, 2) joint list,
the energy volume count and a sequence of (i, j) pairs (either order);
returns a list with, per pair, the row of `joints` between i and j, or -1
where the two are not neighbours in the grid.
"""
from __future__ import annotations

import numpy as np


def joint_index(joints, count: int, pairs) -> list[int]:
    """Index of the joint between each pair of energy volumes, -1 where there is none."""
    if not len(pairs):
        return []
    joints = np.sort(np.asarray(joints, dtype=np.int64).reshape(-1, 2), axis=1)
    wanted = np.sort(np.asarray(pairs, dtype=np.int64).reshape(-1, 2), axis=1)
    wanted = wanted[:, 0] * count + wanted[:, 1]
    if not len(joints):
        return [-1] * len(wanted)
    keys = joints[:, 0] * count + joints[:, 1]
    order = np.argsort(keys)
    place = np.minimum(np.searchsorted(keys[order], wanted), len(keys) - 1)
    found = keys[order][place] == wanted
    return np.where(found, order[place], -1).tolist()
