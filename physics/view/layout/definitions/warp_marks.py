"""Warp marks: the fixed numbers of the layout's warp.

Contract: constants only, none of them a physics setting.
  LIMIT          a cell is drawn at most this many times narrower than the typical cell
                 of its kind; a wider one at most 1 / (2 LIMIT) of the smaller typical
                 width wider. With both, no edge is ever drawn shorter than
                 1 / (2 LIMIT) of its regular length (`cell_layout`)
  REACH          lattice places over which the lines round a smaller cell bend toward
                 it; beyond that nothing moves
  ACROSS         how a line shares the sizes of the two rows on either side of it
  SAMPLE         about how many cells the typical lengths are taken from
  EASE_TURNS     halvings of the search for the largest warp that folds no block
  LAID_OUT_MOST  lattice places laid out at once for the whole grid's blocks
"""
from __future__ import annotations

LIMIT = 4.0
REACH = 8
ACROSS = (0.125, 0.375, 0.375, 0.125)
SAMPLE = 200_000
EASE_TURNS = 12
LAID_OUT_MOST = 2_000_000
