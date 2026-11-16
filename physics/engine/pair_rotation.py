"""Pair rotation (Warp func): the exact two-box fluid step that hands whole quanta from one box to another.

Contract: `pair_rotation(near_amount, far_amount, near_angle, far_angle, turn, uniform)`
takes two boxes' quanta (amounts, ignoring sign, each at least 1), their phases and
the turn -t tau of the pair's own equation H = t/2 [[1, -1], [-1, 1]] (De Raedt 1987):
the complex numbers a = sqrt(n) e^{i phase} are rebuilt, the mean stays and the half
difference turns by `turn`, and the real flux |a|^2 - |a'|^2 = 2 Re(conj(mean) (half -
half')) becomes floor(flux + uniform) whole quanta (uniform in [0, 1): unbiased
rounding, no carry), clamped so neither box drops below 1 quantum. Returns (moved,
near_phase, far_phase): the quanta moved from near to far (negative: far to near) and
the two boxes' new phases arg(a'), unwrapped. The pair's amount is kept exactly by the
caller adding `moved` to far and taking it from near.
"""
from __future__ import annotations

import warp as wp


@wp.func
def pair_rotation(near_amount: int, far_amount: int, near_angle: float, far_angle: float, turn: float,
                  uniform: float):
    """Whole quanta moved near to far by one exact pair rotation, and the two new phases."""
    near_root = wp.sqrt(float(near_amount))
    far_root = wp.sqrt(float(far_amount))
    near_psi = wp.vec2(near_root * wp.cos(near_angle), near_root * wp.sin(near_angle))
    far_psi = wp.vec2(far_root * wp.cos(far_angle), far_root * wp.sin(far_angle))
    mean = (near_psi + far_psi) * 0.5
    half = (near_psi - far_psi) * 0.5
    turned = wp.vec2(half[0] * wp.cos(turn) - half[1] * wp.sin(turn), half[0] * wp.sin(turn) + half[1] * wp.cos(turn))
    change = half - turned
    flux = 2.0 * (mean[0] * change[0] + mean[1] * change[1])
    moved = int(wp.floor(flux + uniform))
    moved = wp.min(moved, near_amount - 1)
    moved = wp.max(moved, -(far_amount - 1))
    near_after = mean + turned
    far_after = mean - turned
    return moved, wp.atan2(near_after[1], near_after[0]), wp.atan2(far_after[1], far_after[0])
