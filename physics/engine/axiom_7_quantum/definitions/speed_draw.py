"""Speed draw: the speed of a split's own oscillation, set at the split, known to nobody.

Contract: `speed_draw(uniform, slowest_speed, fastest_speed)` (host numpy) takes draws in
[0, 1) and returns speeds in [slowest_speed, fastest_speed], in turns per step,
log-uniform: every doubling of speed is as likely as any other. The draw is the split's
own (`SPLIT_SPEED`, of the step it split at and its volume), so the speed is set at the
split. The caller uses it and drops it: no table, count or check ever holds it.
"""
from __future__ import annotations

import math

import numpy as np


def speed_draw(uniform, slowest_speed: float, fastest_speed: float) -> np.ndarray:
    """Speeds between the slowest and the fastest, log-uniform in the draws."""
    if not 0.0 < slowest_speed <= fastest_speed:
        raise ValueError("the speed range must be above 0 and not run backwards")
    low, high = math.log(slowest_speed), math.log(fastest_speed)
    return np.exp(low + (high - low) * np.asarray(uniform, dtype=np.float64))
