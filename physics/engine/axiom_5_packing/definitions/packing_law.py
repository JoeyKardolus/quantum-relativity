"""Packing law: how packed a cell is.

Contract: `packing_law(cell_quanta, length)` (Warp func; int64, float64) returns the
cell's quanta over its length cubed as float64, one law for energy volumes and joints.
0 for an empty cell (no quanta, or endlessly long). Since length is constant / quanta,
packing goes as the quanta to the fourth power: more packed is smaller (axiom 5).
`packing_law_host` is the host twin.
"""
from __future__ import annotations

import math

import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def packing_law(cell_quanta: wp.int64, length: wp.float64) -> wp.float64:
    """Quanta over length cubed; 0 for an empty cell."""
    if cell_quanta <= wp.int64(0) or not (length > wp.float64(0.0)) or wp.isinf(length):
        return wp.float64(0.0)
    return wp.float64(cell_quanta) / (length * length * length)


def packing_law_host(cell_quanta: int, length: float) -> float:
    """Quanta over length cubed; 0 for an empty cell (host)."""
    if int(cell_quanta) <= 0 or not length > 0.0 or math.isinf(length):
        return 0.0
    return int(cell_quanta) / (float(length) * float(length) * float(length))
