"""Read ways: the living ways as host rows, for the viewer.

Contract: `read_ways(state)` returns an int32 numpy array of shape (ways, 4), one row
`(slot, a, b, joint)` per living way in slot order: its slot in the way table, its two
energy volumes (a the lower id) and the joint it goes through. A slot holding no way
(`way_joint = -1`) is left out. It writes nothing.
"""
from __future__ import annotations

import numpy as np


def read_ways(state) -> np.ndarray:
    """The living ways as rows `(slot, a, b, joint)`."""
    joint = state.host("way_joint")
    slots = np.flatnonzero(joint >= 0)
    return np.stack([slots, state.host("way_a")[slots], state.host("way_b")[slots], joint[slots]],
                    axis=1).astype(np.int32)
