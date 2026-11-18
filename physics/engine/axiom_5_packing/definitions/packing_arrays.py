"""Packing arrays: the state axiom 5 owns, how packed each cell is, its clock, the pull and the press.

Contract: `ARRAYS = {name: (dtype, shape, each_step)}` and two host values: `pull_unrest`, the
largest move of the last sweep that set the pull at rest (`rest_pull`), and
`balance_off`, the largest share any joint still moved by in the last round that laid
the start in balance (`new_universe`).
Per cell id:
  packing       float64: quanta over length cubed (packing_law.py)
  rate          float32 in (0, 1]: how fast the cell's local clock runs (clock_rate.py)
  clock         float64: the local time the cell has lived; a ledger, no rule reads it
  pull          float64: the one pull field, energy per plus quantum, deeper is more
                negative; only differences of it are ever used. float64, so a change of
                one part in a million of a cell's energy is still heard (in float32 a
                small change was rounded away at every step)
  pull_old      float64: the pull one step ago (the wave needs it)
  pull_new      float64, cleared each step: the wave's working copy
  pull_made     float32, cleared each step: the source this cell put into the pull this
                step (`spread_pull`); only plus energy makes any
  pull_felt     float32, cleared each step: the largest turn, in radians, the pull gave
                any box of this cell this step (`pull_fluid`); only plus energy feels
                any. Above pi the turn has wrapped and its direction is lost
  press         float32, cleared each step, per joint: its hardest way's press
  squeeze       float32, cleared each step, per joint: press over its own push
  squeezed_way  int32, cleared each step, per joint: the slot of its hardest way.
                `press_joints` writes every living joint, -1 for a joint with no way
Per way slot:
  way_press     float32, cleared each step: the press along this way
"""
from __future__ import annotations

ARRAYS = {
    "packing": ("float64", "cell", False),
    "rate": ("float32", "cell", False),
    "clock": ("float64", "cell", False),
    "pull": ("float64", "cell", False),
    "pull_old": ("float64", "cell", False),
    "pull_new": ("float64", "cell", True),
    "pull_made": ("float32", "cell", True),
    "pull_felt": ("float32", "cell", True),
    "way_press": ("float32", "way", True),
    "press": ("float32", "cell", True),
    "squeeze": ("float32", "cell", True),
    "squeezed_way": ("int32", "cell", True),
}

HOST = {
    "pull_unrest": 0.0,
    "balance_off": 0.0,
}
