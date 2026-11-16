"""Seams: where two joints meet, read off the cells (cells.py), on the host.

Contract: `seams(found)` takes a cells() result and returns a dict:
  boxes   bool (side, side, side): the joint boxes with a face neighbour in another
          joint (the seam's boxes, on both sides of it)
  faces   int: how many faces lie between two boxes of different joints
  pairs   int64 (n, 2): every pair of joint cell ids (lower first) that meet on a face,
          each pair once, sorted
  between int64 (n, 2, 2): for each pair the energy volumes each of its two joints
          lies between (cells()["between"]), -1 for empty space or none
A joint is the one shared slab between two energy volumes (cells.py), so seams run
along the mortar's edges, where three or more volumes' joints meet: the bible's seam,
where joints meet (foundation and rule 6). Boxes on the grid's walls have no neighbour beyond them.
"""
from __future__ import annotations

import numpy as np

from physics.engine.cells import JOINT


def seams(found: dict) -> dict:
    """The seam boxes, seam faces and meeting joint pairs of one universe's cells."""
    owner = found["owner"]
    joint = np.zeros(owner.shape, dtype=bool)
    occupied = owner >= 0
    joint[occupied] = found["kind"][owner[occupied]] == JOINT
    marked = np.zeros(owner.shape, dtype=bool)
    faces = 0
    pairs = []
    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis] = slice(None, -1)
        high[axis] = slice(1, None)
        low, high = tuple(low), tuple(high)
        meeting = joint[low] & joint[high] & (owner[low] != owner[high])
        marked[low] |= meeting
        marked[high] |= meeting
        faces += int(meeting.sum())
        pairs.append(np.sort(np.stack([owner[low][meeting], owner[high][meeting]], axis=1), axis=1))
    joined = np.unique(np.concatenate(pairs).astype(np.int64), axis=0) if faces else np.zeros((0, 2), dtype=np.int64)
    between = np.asarray(found["between"], dtype=np.int64)
    return {"boxes": marked, "faces": faces, "pairs": joined, "between": between[joined]}
