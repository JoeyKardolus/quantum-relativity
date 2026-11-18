"""Palette: the two hues and the opacity scale's fixed numbers.

Contract: constants only.
  VOLUME_RED, JOINT_BLUE   the hue of an energy volume and of a joint; it never changes
  SOLID        a cell is fully solid at this many times the start's mean packing of its
               kind; the opacity is a straight line from 0 up to it
  WHOLE_POWER  in the whole grid the eye looks through two faces of every cell on the
               way, so a face is drawn at the opacity to this power: the start's mean
               0.25 becomes 0.005, three times the mean 0.33, four times solid
None is a physics setting and none is ever changed to fit a frame.
"""
from __future__ import annotations

VOLUME_RED = (255, 38, 26)
JOINT_BLUE = (40, 96, 255)
SOLID = 4.0
WHOLE_POWER = 3.8
