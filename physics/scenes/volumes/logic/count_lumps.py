"""Count lumps: how many energy volumes are lumps, and how many the largest joined region of them holds.

Contract: `count_lumps(universe, settings)` returns `{"lumps": int, "largest_region":
int}`. A lump is a living energy volume holding LUMP_FACTOR times
even_volume_quanta(settings) or more. Two lumps are joined when a living way goes
between them (physics.engine read_ways); `largest_region` is the most lumps joined to
each other that way (largest_group): lumps over regions, not single cells. Writes nothing.
"""
from __future__ import annotations

import numpy as np

from physics.engine import ENERGY_VOLUME, Grid, Settings, read_ways
from physics.scenes.run.definitions.even_volume_quanta import even_volume_quanta
from physics.scenes.volumes.definitions.largest_group import largest_group
from physics.scenes.volumes.definitions.lump_factor import LUMP_FACTOR


def count_lumps(universe: Grid, settings: Settings) -> dict:
    """The number of lumps and the size of the largest region of lumps joined through ways."""
    held = universe.host("cell_quanta")
    lump = (universe.host("kind") == ENERGY_VOLUME) & (held >= LUMP_FACTOR * even_volume_quanta(settings))
    ways = read_ways(universe)
    return {"lumps": int(lump.sum()), "largest_region": largest_group(lump, ways[:, 1], ways[:, 2])}
