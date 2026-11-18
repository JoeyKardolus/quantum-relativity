"""Kinds: the numbers the viewer reads a cell's kind by.

Contract: constants only. `ENERGY_VOLUME` (1), `JOINT` (-1) and `NO_CELL` (0: an id not
used yet, or one whose cell is gone) are the engine's own numbers
(physics.engine, axiom 3). They are written again here so the drawing code can be
loaded without the engine; `source/tests/` checks them against the engine's.
"""
from __future__ import annotations

ENERGY_VOLUME = 1
JOINT = -1
NO_CELL = 0
