"""Validate experiment (milestone 1): measurements recover known results.

Module map:
- run.py: `run(seed, out_dir)`, the entry point, returns the result dict.
- dimension.py: (a) ball-growth and spectral dimension of 1D..4D spatial references.
- proper_time.py: (b) longest causal chain vs sqrt(2 density) T in 1+1.
- twin.py: (c) twin paradox, traveller vs stay-at-home longest chain.
- spacetime_dimension.py: (d) Myrheim-Meyer dimension of 1+1 and 3+1.
- _shared.py: the 10% band and the plot helpers.
"""
from physics.experiments.validate.run import run

__all__ = ["run"]
