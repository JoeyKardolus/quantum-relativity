"""Observer cells: the most packed energy volumes, where an astronomer made of energy would sit.

Contract: `observer_cells(universe, count)` returns the ids (int64 array, fullest first,
a tie to the lower id) of the `count` living energy volumes holding the most quanta now
(fewer when fewer live; none for count <= 0). A cell id is the same cell for good and
the engine's `clock` is that cell's own local time, so an observer's clock is read
straight from the engine. Writes nothing.
"""
from __future__ import annotations

import numpy as np

from physics.engine import ENERGY_VOLUME, Grid


def observer_cells(universe: Grid, count: int) -> np.ndarray:
    """Ids of the `count` energy volumes with the most quanta, fullest first."""
    held = universe.host("cell_quanta").astype(np.float64)
    volumes = np.flatnonzero(universe.host("kind") == ENERGY_VOLUME)
    order = volumes[np.argsort(-held[volumes], kind="stable")]
    return order[:max(int(count), 0)].astype(np.int64)
