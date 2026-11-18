"""Passes: one number per use of a random draw, so no two uses share their draws.

Contract: constants only, the `pass_id` handed to draw_uniform.py.
  START_VOLUME_A, START_VOLUME_B   the two draws of a box's normal at the start (volume or joint box)
  START_SWIRL_AXIS, START_SWIRL_SENSE  a cell's swirl axis and sense at the start
  START_FIELD_A, START_FIELD_B     the two draws of a volume's normal in the start's uneven field
  FLOW_ROUND     rounding inside a cell: add `axis * 2 + parity`, so 10 to 15
  WAY_ROUND      rounding what one facing pair of boxes trades across a way
  WAY_FRICTION   rounding how much of what one pair traded loses its direction in the joint
No phase is ever drawn: the engine has no random angle.
  MAKE_ROUND     rounding what a squeezed joint makes
  SPLIT_SPEED    the speed of a split's own oscillation, drawn at the split (axiom 7)
  SPLIT_POINTER  where a split's pointer jumps
  SETTLE_DEAL    which box gives or gets the first of the quanta left over at a settling
  SPLIT_CARRY    which of a cell's quanta are the ones it sent, for its halves (one draw a cell)
  SPLIT_DEAL     how the ring of the quanta a cell sent is turned against its takers (one draw a cell)
  SPLIT_COOL     rounding how much of a half's direction friction took
"""
from __future__ import annotations

START_VOLUME_A = 1
START_VOLUME_B = 2
START_SWIRL_AXIS = 5
START_SWIRL_SENSE = 6
START_FIELD_A = 7
START_FIELD_B = 8
FLOW_ROUND = 10
WAY_ROUND = 20
WAY_FRICTION = 21
MAKE_ROUND = 30
SPLIT_SPEED = 40
SPLIT_POINTER = 41
SETTLE_DEAL = 42
SPLIT_CARRY = 43
SPLIT_DEAL = 44
SPLIT_COOL = 45
