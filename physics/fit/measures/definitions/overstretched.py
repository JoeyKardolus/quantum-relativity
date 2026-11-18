"""Overstretched: a cell longer than the tear length.

Contract: `overstretched(length, tear_length)` returns a bool array: True where a cell's
length is above the engine's `tear_length` setting. Overstretched means too big, and a
cell is big because it holds too little energy (the size law). A cell holding nothing is
endlessly long, so it is overstretched too. A cell exactly at the tear length is not.
"""
from __future__ import annotations

import numpy as np


def overstretched(length: np.ndarray, tear_length: float) -> np.ndarray:
    """Per cell: longer than the tear length."""
    return np.asarray(length) > tear_length
