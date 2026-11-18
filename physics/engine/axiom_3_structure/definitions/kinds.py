"""Kinds: the two kinds of cell and the size of a cell's block of compute boxes.

Contract: constants only. A cell id is an energy volume (`ENERGY_VOLUME`, its boxes hold
plus quanta or 0), a joint (`JOINT`, its boxes hold minus quanta or 0) or no cell
(`NO_CELL`: an id not yet used, or one whose cell is gone; every box of it holds 0 and
the id is never used again). A cell's kind never changes while it lives (axiom 3: the
pattern never flickers). `BLOCK` is the number of Planck boxes along one edge of a
cell's block and `BOXES` the number in the block. The boxes are not space (axiom 3):
they sit inside the cell only to compute its fluid.
"""
from __future__ import annotations

ENERGY_VOLUME = 1
JOINT = -1
NO_CELL = 0

BLOCK = 6
BOXES = BLOCK ** 3
