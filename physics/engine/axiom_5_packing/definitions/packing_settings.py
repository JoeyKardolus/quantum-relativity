"""Packing settings: the numbers axiom 5 leaves open.

Contract: `PackingSettings` is a frozen record; every default was chosen before any run.
  light_speed    1.0     the speed the pull spreads at; nothing else reads it
  pull_strength  1e-7    G: the pull's source is 4 pi G times a volume's packing
  push_strength  2e-6    P: a joint pushes back with P (its quanta / its length) squared
  clock_packing  6480.0  the packing at which a clock runs at half rate (ten times the
                         packing of the old start's mean volume; kept as a setting for
                         the fit)
`check_packing()` raises ValueError when one of them is not above 0 (no setting is an
off switch); the record runs it when it is made, and the engine's `Settings` runs it
again.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PackingSettings:
    """Light speed, the pull's and the push's strength, and the clock's packing."""

    light_speed: float = 1.0
    pull_strength: float = 1e-7
    push_strength: float = 2e-6
    clock_packing: float = 6480.0

    def __post_init__(self) -> None:
        self.check_packing()

    def check_packing(self) -> None:
        """Refuse any of the four that is not above 0."""
        for name in ("light_speed", "pull_strength", "push_strength", "clock_packing"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be above 0, got {getattr(self, name)}")
