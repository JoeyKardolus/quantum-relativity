"""Volume energy: the energy volumes' quanta as a field over the start lattice.

Contract: `volume_energy(universe)` returns a float64 array of shape (n, n, n), n =
side / 2: what the energy volume laid at lattice place (2 i, 2 j, 2 k) holds now, 0
where that cell is no longer an energy volume (it fell into another). Reads `kind` and
`cell_quanta`; changes nothing.
"""
from __future__ import annotations

import numpy as np

from physics.engine import ENERGY_VOLUME, Grid


def volume_energy(universe: Grid) -> np.ndarray:
    """What every start volume holds now, by its place in the start lattice."""
    side = universe.side
    cells = np.arange(side ** 3).reshape(side, side, side)[::2, ::2, ::2]
    kind, quanta = universe.host("kind"), universe.host("cell_quanta")
    return np.where(kind[cells] == ENERGY_VOLUME, quanta[cells], 0).astype(np.float64)
