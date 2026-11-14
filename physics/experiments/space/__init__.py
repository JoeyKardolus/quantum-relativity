"""Space test (milestone 5): does a network with no geometry settle into 3D space?

Module map:
  run.py                 run: the experiment; plots into out_dir, returns the result dict
  high_dimensional.py    high_dimensional_reference: k-nearest network on a d-torus, d > 4
  track.py               track: steps a network under the rules, measures dimension over time
  plots.py               plot_tracks: the dimension-vs-time and steps-across PNGs
"""
from physics.experiments.space.run import run

__all__ = ["run"]
