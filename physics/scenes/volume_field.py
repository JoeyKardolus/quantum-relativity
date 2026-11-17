"""Volume field: the energy volumes' quanta laid out on the lattice they started on.

Contract: `volume_field(universe)` returns float64 (side / 2, side / 2, side / 2): entry
[i, j, k] is the quanta of the energy volume at lattice place (2i, 2j, 2k), 0 where that
volume is gone (it fell into another, which then holds its quanta). Volumes a tear made
have no lattice place and are left out; `left_out` in the second return value counts
their quanta, so a caller can say how much the field misses. Returns (field, left_out).
"""
from __future__ import annotations

import numpy as np

from physics.engine import ENERGY_VOLUME, State


def volume_field(universe: State) -> tuple[np.ndarray, int]:
    """Quanta of every lattice energy volume, on the volumes' own grid, and the quanta of volumes with no place."""
    side = universe.side
    held = universe.host("cell_quanta").astype(np.float64)
    kind = universe.kind_h[:universe.count]
    held[kind != ENERGY_VOLUME] = 0.0
    field = held[:universe.lattice].reshape(side, side, side)[::2, ::2, ::2]
    return field.copy(), int(held[universe.lattice:].sum())
