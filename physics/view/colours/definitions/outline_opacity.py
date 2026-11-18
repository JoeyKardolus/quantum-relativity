"""Outline opacity: how solid the cells' outlines are drawn, from how large a cell is on screen.

Contract: `outline_opacity(pixels_per_place)` returns a float in 0 to OUTLINE: OUTLINE
when a lattice place is OUTLINE_FULL_AT pixels wide or wider, less in proportion when
narrower, 0 at 0. The pattern is always drawn; seen from far away it fades instead of
filling the picture.
"""
from __future__ import annotations

from physics.view.colours.definitions.palette import OUTLINE, OUTLINE_FULL_AT


def outline_opacity(pixels_per_place: float) -> float:
    """The outlines' opacity for cells this many pixels wide."""
    return OUTLINE * min(max(float(pixels_per_place) / OUTLINE_FULL_AT, 0.0), 1.0)
