"""Joint push: the force with which each joint swells back toward its rest thickness (bible rule 8).

Contract: `joint_push(thickness, rest, push, power)` returns, per joint (t thickness,
R rest),
  push * R * ((R/t)^power - 1) / power
zero at rest, slope -push there (so a small swing keeps the period of a straight spring
of strength push), positive below rest and negative above it. For power > 0 it grows
without bound as t falls to 0, rising steeply like a hard wall, so ordinary pull never
squeezes a joint far and only huge pull reaches the push limit. power -1 is the straight
spring push * (R - t); power 0 is the limit push * R * ln(R/t). A thickness at or below
MIN_SHARE * R is read as MIN_SHARE * R, so the push stays finite; a joint with rest 0
pushes nothing. Moves no energy.
"""
from __future__ import annotations

import numpy as np

MIN_SHARE = 1e-3


def joint_push(thickness: np.ndarray, rest: np.ndarray, push: float, power: float) -> np.ndarray:
    """Per joint, push * R * ((R/t)^power - 1) / power (its log limit at power 0)."""
    held = np.maximum(thickness, MIN_SHARE * rest)
    log_ratio = np.log(np.divide(rest, held, out=np.ones_like(held, dtype=float), where=held > 0))
    if power == 0:
        return push * rest * log_ratio
    return push * rest * np.expm1(power * log_ratio) / power
