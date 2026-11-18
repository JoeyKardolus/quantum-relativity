"""Stat names: the counts a source hands the title.

Contract: constants only. `STAT_NAMES` in the order the title shows them:
  volumes, joints   living cells by kind
  collapses, tears  events since the start
  endless           living cells holding no energy (endlessly long by the size law)
  size              the sum of the lengths of every living cell that holds energy
  size_growth       size over the start's size
  amount            every quantum counted, plus and minus as sizes, exact
  warp              the engine's `measure_grid` warp: below 1 the grid is pulled
                    together round its lumps
"""
from __future__ import annotations

STAT_NAMES = ("volumes", "joints", "collapses", "tears", "endless", "size", "size_growth", "amount", "warp")
