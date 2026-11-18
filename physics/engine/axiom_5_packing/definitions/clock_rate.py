"""Clock rate: more packed is slower.

Contract: `clock_rate(packing, clock_packing)` (Warp func; float64, float64) returns the
rate of a cell's local clock as float32, `1 / (1 + packing / clock_packing)`: 1 for an
empty cell, 1/2 at `clock_packing`, never 0 and never above 1. One law for energy
volumes and joints. Everything in a cell happens in its local time, `step_time * rate`
per step.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def clock_rate(packing: wp.float64, clock_packing: wp.float64) -> wp.float32:
    """How fast a cell's clock runs, in (0, 1]."""
    return wp.float32(wp.float64(1.0) / (wp.float64(1.0) + packing / clock_packing))
