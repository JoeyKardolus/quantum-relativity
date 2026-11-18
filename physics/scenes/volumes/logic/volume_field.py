"""Volume field: the energy volumes' quanta laid out on the lattice they started on.

Contract: `volume_field(universe)` returns `(field, left_out)`. `field` is float64 of
shape (side / 2, side / 2, side / 2): entry [i, j, k] is the quanta of the energy volume
that started at lattice place (2i, 2j, 2k), 0 where that volume is gone (it fell into
another, which then holds its quanta). Volumes a tear made have no lattice place and
are left out; `left_out` is their quanta as an int, so a caller can say how much the
field misses. Reads `cell_quanta` and `kind`; writes nothing.
"""
from __future__ import annotations

import numpy as np

from physics.engine import ENERGY_VOLUME, Grid


def volume_field(universe: Grid) -> tuple[np.ndarray, int]:
    """Quanta of every start energy volume on the volumes' own grid, and the quanta of volumes with no place."""
    side = universe.side
    lattice = side ** 3
    held = universe.host("cell_quanta").astype(np.float64)
    held[universe.host("kind") != ENERGY_VOLUME] = 0.0
    field = held[:lattice].reshape(side, side, side)[::2, ::2, ::2]
    return field.copy(), int(held[lattice:].sum())
