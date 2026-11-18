"""Lump: one placed lump of a gravity scene, as numbers.

Contract: `Lump(centre, width, peak, kick)` is a frozen record.
  centre  (x, y, z) in boxes along the start lattice, a volume being 6 boxes wide: where
          the lump is fullest (a place in the start pattern only; no cell has a place)
  width   in boxes: every box of every energy volume is filled `1 + peak * exp(-r ** 2 /
          (2 width ** 2))` times, r its distance from the centre the short way round
  peak    how many times the even fill the lump adds at its centre; above 0
  kick    (kx, ky, kz) in radians a box: the phase step that sets the lump moving the
          way the vector points (`lumps_start`); (0, 0, 0) is a lump at rest
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lump:
    """Where a placed lump is, how wide and full, and how it is set moving."""

    centre: tuple
    width: float
    peak: float
    kick: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.width > 0.0 or not self.peak > 0.0:
            raise ValueError(f"a lump's width and peak are above 0, got {self.width} and {self.peak}")
