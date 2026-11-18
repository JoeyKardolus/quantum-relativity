"""Energy arrays: the state axiom 1 owns, the whole signed quanta and the books' counts.

Contract: `ARRAYS = {name: (dtype, shape, each_step)}` and `HOST = {name: default}`.
  quanta        int32 per box: plus or 0 in an energy volume, minus or 0 in a joint, 0 in
                every box of an id that is no cell. Zero is empty and does nothing.
  cell_quanta   int64 per cell: the sum of its boxes ignoring sign, only ever written by
                `count_cells`
  moved_inside  int64 per cell, cleared each step: quanta moved between boxes of the
                cell this step
  cut_moves     int32 per cell, cleared each step: moves cut at 0 this step
Host counts since the start, whole numbers (the total is not fixed, every change is
booked; book_law.py):
  made_total     minus quanta squeezed joints made out of themselves
  taken_joint    quanta of collapsed joints, gone with them
  taken_volumes  plus quanta collapses took off the merged volumes
  short_takes    collapses whose merged volume held less than the joint's quanta
  short_quanta   the quanta those collapses could not take: the joint's quanta less
                 what its merged volume held
  torn           plus quanta that tears turned into minus
  start_books    the books as read at the start (None until `new_universe` reads them)
"""
from __future__ import annotations

ARRAYS = {
    "quanta": ("int32", "box", False),
    "cell_quanta": ("int64", "cell", False),
    "moved_inside": ("int64", "cell", True),
    "cut_moves": ("int32", "cell", True),
}

HOST = {
    "made_total": 0,
    "taken_joint": 0,
    "taken_volumes": 0,
    "short_takes": 0,
    "short_quanta": 0,
    "torn": 0,
    "start_books": None,
}
