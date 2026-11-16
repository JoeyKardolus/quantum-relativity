"""Near: which boxes of one universe have a marked box among their neighbours, on the host.

Contract: `near(marked, wrap, faces_only=False, wall=False)` takes a bool array (side,
side, side) and returns a bool array of the same shape: True where a box has a marked
box among its 26 neighbours (or its 6 face neighbours with faces_only), the box itself
not counted. With wrap the universe wraps round at its edges; without it a place past
the walls counts as marked when `wall` is True (a wall is no space), else as unmarked.
Collapse, split and stats read it.
"""
from __future__ import annotations

import itertools

import numpy as np


def near(marked: np.ndarray, wrap: bool, faces_only: bool = False, wall: bool = False) -> np.ndarray:
    """True where a neighbour (26, or 6 across faces) is marked."""
    padded = marked if wrap else np.pad(marked, 1, constant_values=wall)
    found = np.zeros(padded.shape, dtype=bool)
    for offset in itertools.product((-1, 0, 1), repeat=3):
        distance = sum(abs(value) for value in offset)
        if distance == 0 or (faces_only and distance != 1):
            continue
        found |= np.roll(padded, offset, axis=(0, 1, 2))
    return found if wrap else found[1:-1, 1:-1, 1:-1]
