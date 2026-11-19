"""Least whirl: the fewest quanta that are a whirl, and how a torn-off volume shares what gathered.

A cell is a whirl only at `least_whirl` quanta or more: a sixtieth of a calm volume, 50
quanta a box. The number is kept from the whirl before this one, whose emptiest box then
held one whole quantum; a ring whirl fills every box alike and has no emptiest box to
read it off, so it stands as assumed until the owner rules. What has gathered at a free
place is shared in `PARTS` equal parts when it tears off: the new volume's own, and one
toward each of its six sides.
"""
from physics.engine.axiom_3_structure.definitions.kinds import QUANTA_A_BOX

PARTS = 7  # a volume and its six sides: no number of its own
A_SIXTIETH = 60  # of a calm volume: kept, not fitted


def least_whirl(across: int) -> float:
    """The fewest quanta that are a whirl: the same share of a cell at any boxes across."""
    return QUANTA_A_BOX * across ** 3 / A_SIXTIETH
