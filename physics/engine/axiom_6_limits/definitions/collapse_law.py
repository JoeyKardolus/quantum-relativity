"""Collapse law: a joint squeezed too far collapses.

Contract: `collapse_law(joint_quanta, fresh_quanta, collapse_thin)` (Warp func; int64,
int64, float64) returns a bool: `joint_quanta >= collapse_thin * fresh_quanta`, for a
joint that was made with something. A joint's energy rises only while it is squeezed
(make_law.py) and never falls, so what it holds over what it was made with is how far it
has been squeezed, and `collapse_thin` times is too far. It does not ask whether the
joint is squeezed at this moment: a joint past its limit that an earlier event of the
step, or a merge round it, left standing is still past its limit and still collapses.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def collapse_law(joint_quanta: wp.int64, fresh_quanta: wp.int64, collapse_thin: wp.float64) -> bool:
    """Whether a joint has been squeezed to its limit."""
    return fresh_quanta > wp.int64(0) and wp.float64(joint_quanta) >= collapse_thin * wp.float64(fresh_quanta)
