"""Quantum settings: the numbers axiom 7 leaves open, the range a split's speed is drawn from.

Contract: `QuantumSettings` is a frozen record.
  slowest_speed  1.0   the slowest a split's own oscillation may turn, in turns per step
  fastest_speed  64.0  the fastest
The speed of one split is drawn once, at the split, somewhere in this range
(speed_draw.py), and nothing ever returns it: the range is known, the speed is not.
`check_quantum()` raises ValueError when a speed is not above 0 or the fastest is below
the slowest; the record runs it when it is made, and the engine's `Settings` runs it
again.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantumSettings:
    """The range of a split's oscillation speed."""

    slowest_speed: float = 1.0
    fastest_speed: float = 64.0

    def __post_init__(self) -> None:
        self.check_quantum()

    def check_quantum(self) -> None:
        """Refuse a speed that is not above 0 and a range that runs backwards."""
        if not self.slowest_speed > 0.0 or not self.fastest_speed > 0.0:
            raise ValueError("slowest_speed and fastest_speed must be above 0")
        if self.fastest_speed < self.slowest_speed:
            raise ValueError("fastest_speed must not be below slowest_speed")
