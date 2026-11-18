"""Typical lengths: how long the usual energy volume and the usual joint are now.

Contract: `typical_lengths(kind, length, lattice)` reads the start lattice's cells (the
first `lattice` ids), every SAMPLE-th share of them when there are more than SAMPLE,
and returns `(volume, joint)`: the median length of the living energy volumes and of
the living joints that have a length (1.0 for a kind with none). The perfect grid the
drawing starts from is made of these two, so it changes only when the calm background
itself changes size, and then evenly; a lump, however packed, does not move a median.
"""
from __future__ import annotations

import numpy as np

from physics.view.layout.definitions.kinds import ENERGY_VOLUME, JOINT
from physics.view.layout.definitions.warp_marks import SAMPLE


def typical_lengths(kind: np.ndarray, length: np.ndarray, lattice: int) -> tuple:
    """The median length of the energy volumes and of the joints."""
    stride = max(1, lattice // SAMPLE)
    kind = np.asarray(kind)[:lattice:stride]
    length = np.asarray(length)[:lattice:stride].astype(np.float64)
    typical = []
    for one in (ENERGY_VOLUME, JOINT):
        mine = length[(kind == one) & np.isfinite(length) & (length > 0.0)]
        typical.append(float(np.median(mine)) if mine.size else 1.0)
    return typical[0], typical[1]
