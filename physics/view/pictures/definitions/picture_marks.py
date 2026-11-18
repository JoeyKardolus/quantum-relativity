"""Picture marks: the fixed numbers of a saved picture.

Contract: constants only.
  WIDTH, HEIGHT  pixels of a saved picture (a slice's height follows from its shape)
  MARGIN         every picture shows the grid's frame with this share of its width
                 free all round
  PNG_SIGNATURE  the eight bytes every PNG file starts with
"""
from __future__ import annotations

WIDTH, HEIGHT = 1100, 825
MARGIN = 0.04
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
