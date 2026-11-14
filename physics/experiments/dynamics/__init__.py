"""Grid dynamics (milestone 2): does each rule move the network the way it says?

Module map:
- run.py: `run(seed, out_dir)`, one small network per rule, returns the result dict.
- checks.py: `rule_checks(seed, pull_profile)`, the per-rule measurements and their known values.
- pull_fade.py: `pull_by_hops(seed)`, extra pull around a packed lump per hop count.
- fade_plot.py: `plot_pull_fade(profile, out_dir)`, that profile as the card's picture.
- plots.py: `plot_drift(seed, out_dir)`, mean length over steps under drift vs (1 + h)^n.
"""
from physics.experiments.dynamics.run import run

__all__ = ["run"]
