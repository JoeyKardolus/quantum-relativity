"""Lump marks: the numbers that say what the fit calls a lump, written down before any search.

Contract: constants only, never fitted.
  LUMP_FACTOR   2: an energy volume is a lump volume when it holds at least this many times
                the start's mean quanta per energy volume
  CENTRES       12: the gap between lumps is measured between at most this many lumps,
                the ones holding the most
  SPREAD_LUMPS  3: the spread of the lump volumes' sizes is read only with at least this
                many lump volumes
"""
from __future__ import annotations

LUMP_FACTOR = 2
CENTRES = 12
SPREAD_LUMPS = 3
