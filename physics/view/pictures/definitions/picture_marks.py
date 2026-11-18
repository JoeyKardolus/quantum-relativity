"""Picture marks: the fixed numbers of a saved picture.

Contract: constants only.
  WIDTH, HEIGHT  pixels of a saved picture (a slice's height follows from its shape)
  MARGIN         every picture of a run shows the start's grid with this share of its
                 width free all round, so a cell drawn elsewhere later has really moved
  PNG_SIGNATURE  the eight bytes every PNG file starts with
"""
from __future__ import annotations

WIDTH, HEIGHT = 1100, 825
MARGIN = 0.04
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
