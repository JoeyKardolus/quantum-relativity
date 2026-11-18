"""Limits settings: the numbers axiom 6 leaves open.

Contract: `LimitsSettings` is a frozen record; every default was chosen before any run.
  squeeze_rate   1.0   how fast a squeezed joint makes minus energy out of itself
                       (make_law.py), per unit of its local time
  collapse_thin  3.0   a squeezed joint collapses when it holds this many times its own
                       fresh quanta, so is that many times thinner than when fresh
  tear_length    20.0  an energy volume longer than this tears (a setting for the fit to search)
`check_limits()` raises ValueError when `squeeze_rate` or `tear_length` is not above 0
or `collapse_thin` is not above 1 (a fresh joint must not be at its limit); the record
runs it when it is made, and the engine's `Settings` runs it again.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LimitsSettings:
    """How fast a joint makes energy, when it collapses and when a volume tears."""

    squeeze_rate: float = 1.0
    collapse_thin: float = 3.0
    tear_length: float = 20.0

    def __post_init__(self) -> None:
        self.check_limits()

    def check_limits(self) -> None:
        """Refuse a rate or length that is not above 0 and a collapse limit that is not above 1."""
        if not self.squeeze_rate > 0.0 or not self.tear_length > 0.0:
            raise ValueError("squeeze_rate and tear_length must be above 0")
        if not self.collapse_thin > 1.0:
            raise ValueError(f"collapse_thin must be above 1, got {self.collapse_thin}")
