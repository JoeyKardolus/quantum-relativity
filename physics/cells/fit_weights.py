"""Fit weights: nudge each cell's power weight so its tessellated volume moves to its target size.

Contract: `fit_weights(weights, tessellation, cells, target)` returns one new
weight per entry of `cells` (indices into the tessellation), for cells that should have volume `target` (same length as `cells`).
In a power tessellation a cell's volume grows with its weight at the rate
  dV_i / dw_i = sum over its faces of area / (2 * gap)
(gap: the distance between the two working positions across that face), so one
damped Newton step is
  w_i += DAMPING * (target_i - V_i) / (dV_i / dw_i),
each change kept within STEP_LIMIT times the cell's target radius squared, so
a cell is never pushed through empty. DAMPING 0.5, STEP_LIMIT 0.3. A cell with
no faces (empty) has its weight raised by that limit, so it comes back.
More energy gives a smaller target (rest_volume.py), so a lower weight and a
smaller cell. Weights are labels for computing shapes; no rule reads them.
"""
from __future__ import annotations

import numpy as np

from physics.cells.tessellation import Tessellation

DAMPING = 0.5
STEP_LIMIT = 0.3


def fit_weights(weights: np.ndarray, tessellation: Tessellation, cells: np.ndarray,
                target: np.ndarray) -> np.ndarray:
    """Weights of `cells` after one damped Newton step toward volumes `target`."""
    shared = tessellation.face_joint >= 0
    gap = tessellation.pair_gap[tessellation.face_joint[shared]]
    rate = np.bincount(tessellation.face_cell[shared],
                       tessellation.face_area[shared] / (2 * np.maximum(gap, 1e-12)),
                       len(tessellation.volume))[cells]
    limit = STEP_LIMIT * (3 * target / (4 * np.pi)) ** (2 / 3)
    change = np.divide(DAMPING * (target - tessellation.volume[cells]), rate,
                       out=np.array(limit, dtype=float), where=rate > 0)
    return weights[cells] + np.clip(change, -limit, limit)
