"""Palette: the two hues and the fixed numbers of the opacity scale and of the outlines.

Contract: constants only.
  VOLUME_RED, JOINT_BLUE   the hue of an energy volume and of a joint; it never changes
  SOLID        a cell is fully solid at this many times the start's mean packing of its
               kind; the fill is fully clear at the mean and a straight line between
  WHOLE_POWER  in the whole grid the eye looks through two faces of every cell on the
               way, so a face is drawn at the opacity to this power
  OUTLINE          how solid a cell's outline is drawn when its cell is large on screen
  OUTLINE_FULL_AT  pixels a lattice place must be wide for that; narrower, the outline
                   fades in proportion, so millions of cells never add up to a haze
  OUTLINE_INSET    an outline is drawn this share of the way in toward the cell's
                   middle, so neighbours' outlines lie side by side, each in its hue
None is a physics setting and none is ever changed to fit a frame.
"""
from __future__ import annotations

VOLUME_RED = (255, 38, 26)
JOINT_BLUE = (40, 96, 255)
SOLID = 4.0
WHOLE_POWER = 3.8
OUTLINE = 0.75
OUTLINE_FULL_AT = 6.0
OUTLINE_INSET = 0.08
