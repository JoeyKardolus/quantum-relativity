"""Gravity test (milestone 3): do lumps of energy approach, and does joint collapse run away?

Module map:
  run.py       run: the whole experiment, plots, picture and result dict
  lumps.py     add_lumps: put extra energy on the connections around chosen joints
  history.py   lump_history: step the rules and track both lumps through joint collapse
  heavy.py     heavy_lump_run: one heavy lump past the push limit, joints left per step
  tracking.py  track_lumps: lump centres, widths and distances read from the network
  curves.py    lump_curves: size-relative and fixed-ruler distances per step
  result.py    gravity_result: verdict rule, numbers, summary and meaning
  bending.py   path_closest_approach: how near a shortest path passes a joint
  plots.py     plot_curves: one PNG of several curves over time
"""
from physics.experiments.gravity.run import run

__all__ = ["run"]
