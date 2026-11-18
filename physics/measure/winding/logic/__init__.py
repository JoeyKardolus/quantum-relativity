"""Winding, logic: what happens when the winding is read.

  phase_winding.py  phase_winding: whole turns of phase round every unit square of boxes
  cell_turns.py     cell_turns: one cell's whole turns round an axis, layer by layer
"""
from physics.measure.winding.logic.cell_turns import cell_turns
from physics.measure.winding.logic.phase_winding import phase_winding

__all__ = ["phase_winding", "cell_turns"]
