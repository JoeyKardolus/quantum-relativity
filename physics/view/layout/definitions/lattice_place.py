"""Lattice place: where a slot sits in the start lattice.

Contract: `lattice_place(slot, side)` takes slot numbers (the ids of the start's cells,
0 to side ** 3 - 1) and returns int64 (n, 3): the place (i, j, k) with
slot = (i * side + j) * side + k. A place whose three indices are all even is an energy
volume's; one odd index a face joint's, two an edge joint's, three a corner joint's.
"""
from __future__ import annotations

import numpy as np


def lattice_place(slot: np.ndarray, side: int) -> np.ndarray:
    """The (i, j, k) of every slot number."""
    slot = np.asarray(slot, dtype=np.int64)
    return np.stack([slot // (side * side), (slot // side) % side, slot % side], axis=-1)
