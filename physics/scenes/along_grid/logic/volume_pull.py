"""Volume pull: the pull at every start energy volume, laid out on the volumes' own grid.

Contract: `volume_pull(universe)` returns float64 (side / 2, side / 2, side / 2): entry
[i, j, k] is the engine's `pull` (energy per plus quantum, deeper is more negative) at
the cell that started at lattice place (2i, 2j, 2k). Only differences of it mean
anything. Reads `pull`; writes nothing.
"""
from __future__ import annotations

import numpy as np

from physics.engine import Grid


def volume_pull(universe: Grid) -> np.ndarray:
    """The pull at every start energy volume, on the volumes' own grid."""
    side = universe.side
    return universe.host("pull")[:side ** 3].astype(np.float64).reshape((side,) * 3)[::2, ::2, ::2].copy()
