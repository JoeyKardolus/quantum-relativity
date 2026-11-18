"""Lump volumes: the energy volumes that hold enough to be called part of a lump.

Contract: `lump_volumes(kind, held, start_volume_quanta)` returns a bool array, one per
cell id: True for an energy volume whose quanta are at least `LUMP_FACTOR` times
`start_volume_quanta`, the start's mean quanta per energy volume. The mark is the
start's, so a lump means the same at every step of a run. A joint is never a lump volume.
"""
from __future__ import annotations

import numpy as np

from physics.engine import ENERGY_VOLUME
from physics.fit.measures.definitions.lump_marks import LUMP_FACTOR


def lump_volumes(kind: np.ndarray, held: np.ndarray, start_volume_quanta: float) -> np.ndarray:
    """Per cell: an energy volume holding at least LUMP_FACTOR times the start's mean."""
    return (kind == ENERGY_VOLUME) & (held >= LUMP_FACTOR * start_volume_quanta)
