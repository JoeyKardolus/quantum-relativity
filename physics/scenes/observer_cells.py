"""Observer cells: where an astronomer sits, the most packed energy volumes (timescape's dense region).

Contract: `observer_cells(universe, count, local)` returns an int64 array of up to `count`
cell ids, most packed first: living energy volumes (kind A, alive, at least one box) whose
local-time column in `local` (clocked_run.py, shape (samples, cells)) is finite at every
sample, i.e. that existed and kept a clock for the whole run, ranked by energy per box
(cell_energy / cell_boxes, as the last step or stats left the cell sums). The observer is
picked at the end, as an astronomer lives in today's densest lump, and its clock is
followed back over the run. Empty when no cell qualifies. Raises ValueError for count < 1.
"""
from __future__ import annotations

import numpy as np

from physics.potts import ENERGY_VOLUME, Universe


def observer_cells(universe: Universe, count: int, local: np.ndarray) -> np.ndarray:
    """The `count` most packed living energy volumes that kept a clock for the whole run."""
    if count < 1:
        raise ValueError(f"need count >= 1, got {count}")
    boxes = universe.cell_boxes.numpy().astype(np.float64)
    energy = universe.cell_energy.numpy()
    kind = universe.kind.numpy()
    alive = universe.alive.numpy()
    width = min(local.shape[1], boxes.size)
    whole_run = np.zeros(boxes.size, dtype=bool)
    whole_run[:width] = np.isfinite(local[:, :width]).all(axis=0)
    eligible = (kind == ENERGY_VOLUME) & (alive == 1) & (boxes > 0) & whole_run
    packing = np.where(eligible, energy / np.maximum(boxes, 1.0), -np.inf)
    ranked = np.argsort(-packing, kind="stable")
    return ranked[:count][eligible[ranked[:count]]].astype(np.int64)
