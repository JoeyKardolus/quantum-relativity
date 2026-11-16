"""Observer cells: where an astronomer sits, the most packed energy volumes (timescape's dense region).

Contract: `observer_cells(universe, count, local)` returns an int64 array of up to `count`
energy volume labels of a physics.engine State's universe 0 now (physics.engine cells),
most packed first: volumes whose local-time column in `local` (clocked_run.py, shape
(samples, columns); column v is the volume labelled v at the end) is finite at every
sample, i.e. that existed and kept a clock for the whole run, ranked by whole quanta per
box. The observer is picked at the end, as an astronomer lives in today's densest lump,
and its clock is followed back over the run. Empty when no volume qualifies. Raises
ValueError for count < 1.
"""
from __future__ import annotations

import numpy as np

from physics.engine import State, cells, host_boxes


def observer_cells(universe: State, count: int, local: np.ndarray) -> np.ndarray:
    """The `count` most packed energy volumes that kept a clock for the whole run."""
    if count < 1:
        raise ValueError(f"need count >= 1, got {count}")
    found = cells(host_boxes(universe))
    volumes = min(found["volumes"], local.shape[1])
    whole_run = np.isfinite(local[:, :volumes]).all(axis=0)
    packing = found["quanta"][:volumes] / np.maximum(found["boxes"][:volumes], 1)
    packing = np.where(whole_run, packing, -np.inf)
    ranked = np.argsort(-packing, kind="stable")[:count]
    return ranked[whole_run[ranked]].astype(np.int64)
