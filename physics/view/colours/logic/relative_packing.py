"""Relative packing: every cell's packing as a multiple of the start's mean of its kind.

Contract: `relative_packing(kind, packing, marks)` returns float64 (n,): packing over
`marks[kind]` (`start_marks`) for an energy volume or a joint, 0 for an id that is no
cell. The marks are taken once, at the start, so 1 means "as packed as the start's
mean" in every frame.
"""
from __future__ import annotations

import numpy as np

from physics.view.layout.definitions.kinds import ENERGY_VOLUME, JOINT


def relative_packing(kind: np.ndarray, packing: np.ndarray, marks: dict) -> np.ndarray:
    """Packing over the start's mean packing of the cell's own kind."""
    kind, packing = np.asarray(kind), np.asarray(packing, dtype=np.float64)
    relative = np.zeros(len(kind))
    for one in (ENERGY_VOLUME, JOINT):
        relative[kind == one] = packing[kind == one] / marks[one]
    return relative
