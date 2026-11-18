"""Shown kinds: which kinds of cell a drawing shows.

Contract: constants only. `SHOW_ENERGY_VOLUMES` (1), `SHOW_JOINTS` (2), `SHOW_BOTH` (3);
`SHOWN` is the three together, for checking a value.
"""
from __future__ import annotations

SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOW_BOTH = 1, 2, 3
SHOWN = (SHOW_ENERGY_VOLUMES, SHOW_JOINTS, SHOW_BOTH)
