"""Camera marks: the fixed numbers of the camera.

Contract: constants only.
  FOV_DEGREES   the camera's field of view, top to bottom
  CORNER        (yaw, pitch) the whole grid is first seen from: the corner at lattice
                index 0
  FACING        (yaw, pitch) the slice is seen from: straight along z
  FILL          a fitted grid's widest side fills about this share of half the view
  REFIT_GROWTH  the view is fitted again when the grid outgrows the last fit by this
  LEVELS        depth steps the far-to-near sort cuts the view into
"""
from __future__ import annotations

import math

FOV_DEGREES = 45.0
CORNER = (0.6 + math.pi, -0.35)
FACING = (0.0, 0.0)
FILL = 0.9
REFIT_GROWTH = 1.5
LEVELS = 65535
