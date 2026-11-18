"""Crossing arrays: the state axiom 4 owns, what each way carried and the heat each cell holds.

Contract: `ARRAYS = {name: (dtype, shape, each_step)}`; there is no host value.
Per way slot, all cleared each step:
  way_out_a   int32: whole plus quanta volume a sent through the joint to b this step
  way_out_b   int32: the same from b to a
  way_lost_a  int32: how many of a's lost their direction in the joint (friction)
  way_lost_b  int32: the same of b's
  way_moved   int32: `way_out_a - way_out_b`, the net, a to b above 0
Per cell id:
  crossed        int64, cleared each step, per joint: quanta that crossed through it
  heated         int64, cleared each step, per joint: quanta that lost their direction in it
  quanta_before  int64: `cell_quanta` as it was when this step's crossing began
  heat_quanta    int64, kept: how many of the cell's quanta have had their direction
                 taken by a joint's friction. Only friction makes heat, and heat leaves a
                 cell with the quanta that leave it. It is a count: a box has one phase
                 for all its quanta, so heat moves with the rest of the fluid
  heat_before    int64: `heat_quanta` as it was when this pass of the crossing began
Plus energy is in a box at every moment (axiom 1), and a joint's boxes hold minus, so the
crossing energy is never stored in the joint: there is no array for it.
"""
from __future__ import annotations

ARRAYS = {
    "way_out_a": ("int32", "way", True),
    "way_out_b": ("int32", "way", True),
    "way_lost_a": ("int32", "way", True),
    "way_lost_b": ("int32", "way", True),
    "way_moved": ("int32", "way", True),
    "crossed": ("int64", "cell", True),
    "heated": ("int64", "cell", True),
    "quanta_before": ("int64", "cell", False),
    "heat_quanta": ("int64", "cell", False),
    "heat_before": ("int64", "cell", False),
}

HOST = {}
