"""Size law: a cell's length from its energy alone.

Contract: `size_law(cell_quanta, size_constant)` (Warp func) returns the cell's length,
`size_constant / cell_quanta` worked in float64 and stored as float32: more energy,
smaller. One law and one constant for energy volumes and joints alike (axiom 3). A cell
holding 0 quanta is endlessly long (`inf`). `size_law_host` is the host twin and returns
the same float32 value as a Python float.
"""
from __future__ import annotations

import numpy as np
import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def size_law(cell_quanta: wp.int64, size_constant: wp.float64) -> wp.float32:
    """The length of a cell holding `cell_quanta` quanta."""
    if cell_quanta <= wp.int64(0):
        return wp.float32(wp.inf)
    return wp.float32(size_constant / wp.float64(cell_quanta))


def size_law_host(cell_quanta: int, size_constant: float) -> float:
    """The length of a cell holding `cell_quanta` quanta (host)."""
    if int(cell_quanta) <= 0:
        return float("inf")
    return float(np.float32(np.float64(size_constant) / np.float64(int(cell_quanta))))
