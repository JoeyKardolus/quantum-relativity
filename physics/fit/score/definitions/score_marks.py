"""Score marks: where each part of the score stops counting more, written down before any search.

Contract: constants only. They are not fitted and say only where a part saturates.
  GROWTH_SCALE  0.1  a length's growth (end over start, minus 1)
  REGION_SCALE  1.0  the gain of region contrast
  LUMP_SCALE    0.4  the gain of the lumps' share of the energy
  HELD_SCALE    0.4  the held share above the start's lump share
  SPREAD_SCALE  0.3  the gain of the spread of the lump volumes' sizes
  MIN_SAMPLES   4    the fewest samples a run can be scored on (three thirds need four cuts)
"""
from __future__ import annotations

GROWTH_SCALE = 0.1
REGION_SCALE = 1.0
LUMP_SCALE = 0.4
HELD_SCALE = 0.4
SPREAD_SCALE = 0.3
MIN_SAMPLES = 4
