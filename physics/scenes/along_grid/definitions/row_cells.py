"""Row cells: the ids of the cells along one row of the start lattice, going out from a cell.

Contract: `row_cells(side, core, axis, sign, places)` takes the lattice's cells along an
edge, a start cell id, an axis (0, 1 or 2), a direction (+1 or -1) and a count, and
returns an int64 array of `places + 1` ids: the core and the next `places` cells along
that row, wrapping round. Along a row through an energy volume the cells alternate
volume, joint, volume: every second id is a volume. Ids are the start's,
`(i * side + j) * side + k`. Raises ValueError for another axis or sign.
"""
from __future__ import annotations

import numpy as np


def row_cells(side: int, core: int, axis: int, sign: int, places: int) -> np.ndarray:
    """The core and the `places` cells after it along one lattice row (wrapping)."""
    if axis not in (0, 1, 2) or sign not in (1, -1):
        raise ValueError(f"axis must be 0, 1 or 2 and sign +1 or -1, got {axis} and {sign}")
    weight = side ** (2 - axis)
    index = (int(core) // weight) % side
    return (int(core) + (((index + sign * np.arange(places + 1)) % side) - index) * weight).astype(np.int64)
