"""The score, definitions: its scales and the formula of every part.

  score_marks.py      the scales where each part stops counting more, and the fewest samples
  gain.py             gain: tanh(change / scale), 0 for a loss
  length_shape.py     length_shape: grew x steady x (1 + dip) / 2 for one length over the observer's time
  structure_parts.py  structure_parts: the four structure gains since the start
  hold_factor.py      hold_factor: alive x (1 - spent share), a straight line with no cliff
"""
