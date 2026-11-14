"""Gravity test (milestone 3): do lumps of energy approach each other with attraction off?

Module map:
  run.py       run: the whole experiment, plots and result dict
  lumps.py     add_lumps: put extra energy on the connections around chosen joints
  tracking.py  track_lumps: lump centres, widths and distances read from the network
  bending.py   path_closest_approach: how near a shortest path passes a joint
  plots.py     plot_curves: one PNG of several curves over time
"""
from physics.experiments.gravity.run import run

__all__ = ["run"]
