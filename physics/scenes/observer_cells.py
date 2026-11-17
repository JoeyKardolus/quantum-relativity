"""Observer cells: the most packed energy volumes, where an astronomer made of energy would sit.

Contract: `observer_cells(universe, count)` returns the ids (int64 array, most packed
first) of the `count` energy volumes holding the most quanta now (fewer when fewer
live). A cell id is the same cell for good and state.clock is that cell's own elapsed
local time, so an observer's clock is read straight from the engine.
"""
from __future__ import annotations

import numpy as np

from physics.engine import ENERGY_VOLUME, State


def observer_cells(universe: State, count: int) -> np.ndarray:
    """Ids of the `count` energy volumes with the most quanta."""
    held = universe.host("cell_quanta").astype(np.float64)
    volumes = np.flatnonzero(universe.kind_h[:universe.count] == ENERGY_VOLUME)
    order = volumes[np.argsort(-held[volumes], kind="stable")]
    return order[:max(int(count), 0)].astype(np.int64)
