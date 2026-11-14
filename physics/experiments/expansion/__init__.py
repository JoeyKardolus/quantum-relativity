"""Milestone 4, expansion test: does drift give accelerating expansion while dense clusters stay bound?

Module map:
  clustered_reference.py  clustered_reference: 3D sparse space with dense clusters (coordinates dropped)
  evolve.py               evolve: step the rules, record scale factor and cluster sizes
  plots.py                draw_plots: the PNG curves
  run.py                  run(seed, out_dir) -> result dict
"""
from physics.experiments.expansion.run import run

__all__ = ["run"]
