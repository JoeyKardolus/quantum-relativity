"""Structure settings: the one number axiom 3 leaves open, the constant of the size law.

Contract: `StructureSettings` is a frozen record.
  size_constant  6.48e6: a cell's length is `size_constant / its quanta` (size_law.py),
                 in length times quanta, the same constant for energy volumes and joints
                 (axiom 3: "for volumes and joints alike"). With the start's 3000 quanta a
                 box a volume is 10 long, with 100 a box a joint is 300 long.
`check_structure()` raises ValueError when the constant is not above 0; the record runs
it when it is made, and the engine's `Settings` runs it again.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructureSettings:
    """The size law's one constant."""

    size_constant: float = 6.48e6

    def __post_init__(self) -> None:
        self.check_structure()

    def check_structure(self) -> None:
        """Refuse a size constant that is not above 0."""
        if not self.size_constant > 0.0:
            raise ValueError(f"size_constant must be above 0, got {self.size_constant}")
