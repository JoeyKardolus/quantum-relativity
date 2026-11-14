"""Expansion rate and its change, from a history of mean connection lengths.

Contract: `expansion_rate(history, step_size)` treats the mean length a(t) as a
scale factor and returns {"rate": H = d ln a / dt, "acceleration": d^2 a / dt^2 / a,
"deceleration": q = -a a'' / a'^2} per step (central differences, numpy.gradient).
Accelerating expansion means acceleration > 0, i.e. q < 0.
"""
from __future__ import annotations

import numpy as np


def expansion_rate(history: np.ndarray | list[float], step_size: float = 1.0) -> dict:
    """Hubble-like rate, relative acceleration and deceleration parameter per step."""
    scale = np.asarray(history, dtype=float)
    speed = np.gradient(scale, step_size)
    push = np.gradient(speed, step_size)
    with np.errstate(divide="ignore", invalid="ignore"):
        deceleration = -scale * push / speed ** 2
    return {"rate": speed / scale, "acceleration": push / scale, "deceleration": deceleration}
