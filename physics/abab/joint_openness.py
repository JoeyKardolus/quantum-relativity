"""Joint openness: how much each joint lets through, 1 relaxed, 0 sealed (bible foundation 4, rule 6).

Contract: `joint_openness(size, rest, settings)` returns one value in [0, 1] per
joint (size t, rest size R from joint_rest_size.py). Squeeze depth
d = max(0, 1 - t / R):
  openness = clip(1 - d / seal, 0, 1),   seal = min(seal_point, 1 - push_limit)
1 at or above rest, falling straight to 0 at the seal point; at and past it the
joint passes nothing. With seal 0 a joint is open only at or above rest.
Raises ValueError for a negative seal_point.
"""
from __future__ import annotations

import numpy as np

from physics.abab.settings import Settings


def joint_openness(size: np.ndarray, rest: np.ndarray, settings: Settings) -> np.ndarray:
    """Per joint, 1 - (its squeeze depth / seal point), clipped to [0, 1]."""
    if settings.seal_point < 0:
        raise ValueError("seal_point must be >= 0: a squeeze depth")
    depth = np.maximum(0.0, 1 - size / rest)
    seal = min(settings.seal_point, 1 - settings.push_limit)
    if seal <= 0:
        return (depth <= 0).astype(float)
    return np.clip(1 - depth / seal, 0.0, 1.0)
