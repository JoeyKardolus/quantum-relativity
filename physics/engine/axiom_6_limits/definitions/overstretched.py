"""Overstretched: when an energy volume tears, where, and when its halves could not be whirls.

A volume is overstretched when its size is above `TEAR_SIZE`. It tears across the direction
it is most stretched along: by its neighbours, as much as it is pulled from both of its
sides along that direction (the smaller of the two sides' pulls; a pull from one side only
stretches nothing); and where those say nothing, by its own fluid, along the direction its
quanta lie widest spread. A half is a whirl only at `least_whirl` quanta or more.
"""
from functools import lru_cache

import numpy as np

from physics.engine.axiom_2_swirls.definitions.swirl_pattern import swirl_patterns
from physics.engine.axiom_3_structure.definitions.kinds import ACROSS

# The one setting, chosen before any run (the fit may search it): twice a calm volume's
# size. A calm volume stands, and the biggest volume a scene has shown standing is the
# crossing scene's emptier one, which starts at exactly 2.
TEAR_SIZE = 2.0


@lru_cache
def least_whirl(across: int) -> float:
    """The fewest quanta that are a whirl: at the least boxes across, a whole quantum in the
    emptiest box the whirl fills; the same share of a cell at any boxes across."""
    share = swirl_patterns(ACROSS)[0, :, 0]
    return across ** 3 / share[share > 1e-9].min()


def layers(cube: np.ndarray, way: int) -> np.ndarray:
    """The quanta of each layer of a block's boxes across direction `way`."""
    return np.abs(np.moveaxis(cube, way, 0)).reshape(len(cube), -1).sum(axis=1)


def spread(cube: np.ndarray, way: int) -> float:
    """How wide a block's quanta lie along direction `way`: their mean square distance from their middle."""
    held, place = layers(cube, way), np.arange(len(cube))
    middle = (held * place).sum() / held.sum()
    return float((held * (place - middle) ** 2).sum() / held.sum())


def torn_across(cube: np.ndarray, pulls: np.ndarray) -> int:
    """The direction a volume tears across: `pulls` is (3, 2), each direction's near and far side."""
    return max(range(3), key=lambda way: (pulls[way].min(), spread(cube, way)))


def half_round(laid: int, way: int) -> int:
    """How a whirl laid `laid` lies once turned half round about direction `way`."""
    return way + 1 if laid == 0 else 0 if laid == way + 1 else 5 - laid - way
