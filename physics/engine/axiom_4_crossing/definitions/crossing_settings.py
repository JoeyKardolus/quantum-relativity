"""Crossing settings: the one number axiom 4 leaves open, the friction of a joint.

Contract: `CrossingSettings` is a frozen record.
  friction  0.5: the share of the quanta crossing a joint that lose their direction
            there (friction_law.py). One share for every joint; there is no seal and no
            rate setting.
`check_crossing()` raises ValueError when the share is not above 0 or is above 1; the
record runs it when it is made, and the engine's `Settings` runs it again.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CrossingSettings:
    """The share of crossing quanta that lose their direction."""

    friction: float = 0.5

    def __post_init__(self) -> None:
        self.check_crossing()

    def check_crossing(self) -> None:
        """Refuse a friction that is not a share above 0."""
        if not 0.0 < self.friction <= 1.0:
            raise ValueError(f"friction must be above 0 and at most 1, got {self.friction}")
