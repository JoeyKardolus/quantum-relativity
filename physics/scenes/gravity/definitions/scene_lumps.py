"""Scene lumps: the lumps each gravity scene places, written before any run.

Contract: `scene_lumps(name, side)` returns the list of `Lump` of one scene in a start
lattice of `side` cells an edge (so `3 * side` boxes), all at rest unless said:
  lone       one lump in the middle
  two_lumps  two equal lumps `PAIR_APART` volumes apart along the first axis (the other
             way round the box is further, so the pair is not held by its own copies)
  coast      one lump with a kick of `COAST_KICK` radians a box along the first axis
  orbit      a big lump (`BIG_PEAK`) in the middle and a small one `ORBIT_APART`
             volumes from it; `orbit_kick` gives the small one its sideways kick
`WIDTH`, `PEAK`, `BIG_PEAK` are in boxes and in even fills. Raises ValueError for a name
that is no scene or a side with fewer than 12 volumes' room for the pair (below 24).
"""
from __future__ import annotations

from physics.scenes.gravity.definitions.lump import Lump

SCENES = ("lone", "two_lumps", "coast", "orbit")
BOXES_A_VOLUME = 6
WIDTH = 5.0
PEAK = 16.0
BIG_PEAK = 48.0
PAIR_APART = 4
ORBIT_APART = 3
COAST_KICK = 0.5


def scene_lumps(name: str, side: int) -> list:
    """The lumps of one scene, placed round the middle of the start lattice."""
    if name not in SCENES:
        raise ValueError(f"no gravity scene {name!r}; there are {SCENES}")
    if side < 24:
        raise ValueError(f"a gravity scene needs 24 cells a side or more, got {side}")
    middle = (side // 4) * BOXES_A_VOLUME + 2.5
    if name == "lone":
        return [Lump((middle, middle, middle), WIDTH, PEAK)]
    if name == "coast":
        return [Lump((middle - 2 * BOXES_A_VOLUME, middle, middle), WIDTH, PEAK, (COAST_KICK, 0.0, 0.0))]
    if name == "two_lumps":
        half = PAIR_APART * BOXES_A_VOLUME / 2.0
        return [Lump((middle - half, middle, middle), WIDTH, PEAK), Lump((middle + half, middle, middle), WIDTH, PEAK)]
    return [Lump((middle, middle, middle), WIDTH, BIG_PEAK),
            Lump((middle + ORBIT_APART * BOXES_A_VOLUME, middle, middle), WIDTH / 2.0, PEAK)]
