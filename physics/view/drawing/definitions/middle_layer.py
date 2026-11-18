"""Middle layer: the layer the slice starts at.

Contract: `middle_layer(side)` returns the even lattice index nearest the middle of a
lattice of `side` places: energy volumes lie only in even layers (an odd layer is all
joints), so the first slice shows both kinds.
"""
from __future__ import annotations


def middle_layer(side: int) -> int:
    """The even lattice layer nearest the middle."""
    return side // 2 - (side // 2) % 2
