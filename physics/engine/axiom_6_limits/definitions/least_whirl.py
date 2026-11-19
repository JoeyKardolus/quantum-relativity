"""Least whirl: the fewest quanta that are a whirl, and how a torn-off volume shares what gathered.

A cell is a whirl only at `least_whirl` quanta or more. What has gathered at a free place
is shared in `PARTS` equal parts when it tears off: the new volume's own, and one toward
each of its six sides.
"""
from functools import lru_cache

from physics.engine.axiom_2_swirls.definitions.swirl_pattern import swirl_patterns
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS

PARTS = 7  # a volume and its six sides: no number of its own


@lru_cache
def least_whirl(across: int) -> float:
    """The fewest quanta that are a whirl: at the least boxes across, a whole quantum in the
    emptiest box the whirl fills; the same share of a cell at any boxes across."""
    share = swirl_patterns(ACROSS)[0, :, 0]
    return across ** 3 / share[share > 1e-9].min()
