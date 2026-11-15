"""Joint balance: the size where a joint's push cancels the squeeze on it.

Contract: `joint_balance(rest, squeeze, settings)` returns one size per joint, the
point in (0, rest] where swing_joints()'s force is zero,
  joint_push(t, R, k, power) = s * t,  i.e.  k ((R/t)^power - 1) / power = s t / R
found by BISECTIONS (50) halvings of t / R between MIN_SHARE and 1. R when s is 0.
new_state() starts every joint here at rate 0.
"""
from __future__ import annotations

import numpy as np

from physics.abab.settings import Settings
from physics.abab.joint_push import MIN_SHARE, joint_push

BISECTIONS = 50


def joint_balance(rest: np.ndarray, squeeze: np.ndarray, settings: Settings) -> np.ndarray:
    """Per joint, the size in (0, rest] where push and squeeze cancel."""
    ones = np.ones(len(squeeze))
    low, high = np.full(len(squeeze), MIN_SHARE), ones.copy()
    for _ in range(BISECTIONS):
        middle = (low + high) / 2
        pushed_back = joint_push(middle, ones, settings.joint_push, settings.push_power) >= squeeze * middle
        low, high = np.where(pushed_back, middle, low), np.where(pushed_back, high, middle)
    return rest * (low + high) / 2
