"""Make law: a joint's energy rises only while it is squeezed, made out of the joint itself.

Contract: `make_law(joint_quanta, squeeze, squeeze_rate, tau)` (Warp func; int64,
float64, float64, float64) returns as float64 the minus quanta a joint makes in `tau` of
its local time:
  joint_quanta * (squeeze ** (1 / 4) - 1) * (1 - exp(-squeeze_rate * tau))
The first two factors are what would bring its push level with the press (the push goes
as the quanta to the fourth power); the last is how much of that way it gets in `tau`,
going toward the level at `squeeze_rate` per unit of its own time. So it never makes
more than brings it level, a slower clock makes it slower, and `squeeze_rate` always
counts. 0 at or below a squeeze of 1: an unsqueezed joint makes nothing, and nothing
ever takes energy off a living joint, so a fresh joint is the fattest it will ever be.
The caller rounds it to whole quanta.
"""
from __future__ import annotations

import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.func
def make_law(joint_quanta: wp.int64, squeeze: wp.float64, squeeze_rate: wp.float64, tau: wp.float64) -> wp.float64:
    """The minus quanta a squeezed joint makes in `tau` of its local time."""
    if not (squeeze > wp.float64(1.0)) or joint_quanta <= wp.int64(0):
        return wp.float64(0.0)
    level = wp.float64(joint_quanta) * (wp.pow(squeeze, wp.float64(0.25)) - wp.float64(1.0))
    return wp.max(level * (wp.float64(1.0) - wp.exp(-squeeze_rate * tau)), wp.float64(0.0))
