"""Middle distances: how far each cell of a list lies from the first, along the list, in summed lengths.

Contract: `middle_distances(universe, cells)` takes a universe and a sequence of cell
ids that lie one after another (a row, row_cells) and returns a float64 array as long:
entry n is the distance from the middle of cells[0] to the middle of cells[n], the sum
of the lengths of the cells between plus half of each end (the bible's distance: the
sum of the lengths of the cells between). Entry 0 is 0. Reads `length`; writes nothing.
"""
from __future__ import annotations

import numpy as np

from physics.engine import Grid


def middle_distances(universe: Grid, cells) -> np.ndarray:
    """Summed lengths from the first cell's middle to every listed cell's middle."""
    along = universe.host("length").astype(np.float64)[np.asarray(cells, dtype=np.int64)]
    return np.cumsum(along) - 0.5 * along - 0.5 * along[0]
