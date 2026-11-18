"""Point holds: when a searched point still stands in a run four times longer.

Contract: `point_holds(short_score, long_score, long_hold)` returns True when the short
score is above 0, the long run's score is at least `HALF_KEPT` (0.5) of it, and the long
run's hold factor stayed above `CALM` (0.9). A short score can be a runaway caught
midway; only a point that holds may be fitted.
"""
from __future__ import annotations

HALF_KEPT = 0.5
CALM = 0.9


def point_holds(short_score: float, long_score: float, long_hold: float) -> bool:
    """Whether the long run kept half the short score and stayed calm."""
    return short_score > 0.0 and long_score >= HALF_KEPT * short_score and long_hold > CALM
