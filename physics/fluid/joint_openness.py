"""Joint openness: how much a joint lets through, 1 relaxed, 0 at and past the seal point.

Contract: a Warp function, usable inside kernels only. `joint_openness(compression,
seal)` with c the B compression (b / b_rest) and seal the seal compression: squeeze
depth d = 1 - c**(-1/3) (how much thinner than at rest the joint is, its thickness
going as c**(-1/3)), seal depth D = 1 - seal**(-1/3), openness = clip(1 - d / D, 0, 1).
1 at or below rest, falling straight to 0 at the seal point (bible foundation 4 and
rule 6: the more squeezed, the less passes; past the seal nothing). The earlier ABAB
engine's joint_openness in density terms (git history). A seal of inf or more gives
D = 1. 0 for c <= 0 is never asked: a face with no B has no joint (leak.py).
"""
from __future__ import annotations

import warp as wp


@wp.func
def joint_openness(compression: float, seal: float) -> float:
    """1 at rest, straight down to 0 at the seal point's squeeze depth."""
    depth = 1.0 - wp.pow(wp.max(compression, 1.0e-30), -1.0 / 3.0)
    seal_depth = 1.0 - wp.pow(seal, -1.0 / 3.0)
    return wp.clamp(1.0 - depth / seal_depth, 0.0, 1.0)
