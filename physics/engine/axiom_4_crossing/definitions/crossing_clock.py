"""Crossing clock: a crossing runs on the slowest of the three clocks it goes through.

Contract: `crossing_clock(step_time, rate_a, rate_b, rate_joint)` (Warp func, float64)
returns the local time one step is worth on a way: `step_time` times the slowest of the
clock rates of its two volumes and of the joint between them. Energy leaves one volume,
goes through the joint and enters the other, so the slowest of the three sets how much
gets through: a packed joint lets less cross, by its clock and by nothing else.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def crossing_clock(step_time: wp.float64, rate_a: wp.float64, rate_b: wp.float64,
                   rate_joint: wp.float64) -> wp.float64:
    """One step's local time on a way: the slowest of its three clocks."""
    return step_time * wp.min(rate_a, wp.min(rate_b, rate_joint))
