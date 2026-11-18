"""Shell ways: how many ways lie between each energy volume of the start lattice and the core.

Contract: `shell_ways(across, centre)` takes the volumes along an edge of the start
lattice and the core's index along each axis (the same on all three) and returns an int
array (across, across, across): entry [i, j, k] is the fewest ways from the core to
that volume. Every volume has a way to each of the 26 round it (axiom 3: every joint in
the pattern is a real joint), so that is the largest of the three wrapping offsets.
Shell n is where the entry equals n; boundary n closes the core and shells 1 .. n.
Raises ValueError for across < 2 or a centre off the lattice.
"""
from __future__ import annotations

import numpy as np


def shell_ways(across: int, centre: int) -> np.ndarray:
    """Per start volume, the fewest ways to the core (wrapping)."""
    if across < 2 or not 0 <= centre < across:
        raise ValueError(f"need across >= 2 and a centre on the lattice, got {across} and {centre}")
    offset = np.abs(np.indices((across,) * 3) - centre)
    return np.minimum(offset, across - offset).max(axis=0)
