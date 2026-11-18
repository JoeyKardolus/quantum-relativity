"""Fit marks: the fixed numbers of the layout's fit.

Contract: constants only, none of them a physics setting.
  RELATIVE     the solve stops when what is left of the load is this share of it
  MOST_TURNS   turns of the solve before it is given up for the direct one
  SHEAR        how much a miss sideways counts beside a miss in length
  FLOOR        an empty slot is held this share of the median width wide, so the fit
               never divides by nothing
"""
from __future__ import annotations

RELATIVE = 1e-8
MOST_TURNS = 300
SHEAR = 0.3
FLOOR = 1e-3
