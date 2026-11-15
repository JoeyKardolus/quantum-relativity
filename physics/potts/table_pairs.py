"""Table pairs: the touching pairs as the GPU contact table holds them, back on the host.

Contract: `table_pairs(universe)` returns (low, high, faces), three int64 numpy arrays, one
entry per pair in universe.neighbour / faces (contact_table.py): low < high the two cells,
faces their shared box faces, sorted by (low, high). It reads only the per-cell table
(cell_count * slots entries), never the boxes, so joint collapse and volume split can
change the pairs they touch without recounting the whole grid.
"""
from __future__ import annotations

import numpy as np

from physics.potts.universe import Universe


def table_pairs(universe: Universe) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(low, high, faces) from the contact table."""
    neighbour = universe.neighbour.numpy()
    faces = universe.faces.numpy()
    used = np.flatnonzero(neighbour >= 0)
    low = (used // universe.slots).astype(np.int64)
    high = neighbour[used].astype(np.int64)
    order = np.lexsort((high, low))
    return low[order], high[order], faces[used][order].astype(np.int64)
