"""Events: the collapses and tears a step found are carried out, one at a time, and the relations copied back.

Contract: `events(state, settings)` reads state.flags (1: a squeezed joint past its
limit, squeeze.py; 2: an energy volume grown past the tear length, cell_laws.py),
collapses those joints in id order (collapse.py), then tears those volumes in id order
(tear.py), each checked again as it comes (an earlier one may have changed it), on a
host copy of the per-cell arrays that is written back once, with the changed relations
(state.relink). These are the only places a cell's neighbours change. Room for the new
ids of every tear is made first (state.grow). Returns (collapses, tears) carried out.
"""
from __future__ import annotations

import numpy as np

from physics.engine.collapse import collapse
from physics.engine.settings import Settings
from physics.engine.state import State
from physics.engine.tear import tear

_HELD = ("cell_quanta", "pull", "pull_old", "clock", "jiggle")


def events(state: State, settings: Settings) -> tuple:
    """Carry out this step's flagged collapses and tears; returns how many of each happened."""
    flags = state.host("flags")
    joints, volumes = np.flatnonzero(flags == 1), np.flatnonzero(flags == 2)
    if len(joints) == 0 and len(volumes) == 0:
        return 0, 0
    state.grow(state.count + 2 * len(volumes))
    held = {name: getattr(state, name).numpy() for name in _HELD}
    collapsed = sum(collapse(state, held, joint) for joint in joints)
    torn = sum(tear(state, held, volume, settings.tear_length) for volume in volumes)
    for name in _HELD:
        getattr(state, name).assign(held[name])
    state.relink()
    return collapsed, torn
