"""Level: the four detail levels a block of Planck boxes can sit at.

Contract: `Level` is an IntEnum whose values are the codes the kernels read.
  EMPTY   every box holds 0 quanta (no space): no storage at all, only its index entry
  FULL    every box stored: int32 quanta and float32 phase, 8 bytes a box
  COARSE  a quiet block: per group of 4^3 boxes one shared lowest magnitude, step and
          exact remainder per sign, and a shared phase centre and span; per box a 2-bit
          sign and 4-bit magnitude and phase offsets (GSQ-style, arXiv 2604.18556)
  REUSED  a quiet block whose exact content equals another's: one shared FULL copy,
          keyed by its exact bytes (Hashlife-style), held by every block that matches it
"""
from __future__ import annotations

from enum import IntEnum


class Level(IntEnum):
    """Detail level of one block."""

    EMPTY = 0
    FULL = 1
    COARSE = 2
    REUSED = 3
