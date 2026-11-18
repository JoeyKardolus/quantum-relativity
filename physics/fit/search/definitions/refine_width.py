"""Refine width: how wide a refine round steps round the best points.

Contract: `refine_width(round_index)` returns the standard deviation, in unit-cube
units, of the Gaussian steps of refine round 1, 2, ...: `FIRST_WIDTH` (0.15) in round 1
and `NARROWING` (0.7) times narrower each round after. Raises ValueError for a round
below 1.
"""
from __future__ import annotations

FIRST_WIDTH = 0.15
NARROWING = 0.7


def refine_width(round_index: int) -> float:
    """The width of a refine round's steps in the unit cube."""
    if round_index < 1:
        raise ValueError(f"refine rounds count from 1, got {round_index}")
    return FIRST_WIDTH * NARROWING ** (round_index - 1)
