"""Grid stats: what one grid looks like now, read from the grid only.

Contract: `grid_stats(grid, start_total_volume, start_cells=None)` returns a dict:
  cells           energy volumes now
  collapsed       start_cells - cells now (volumes that fell into others so far);
                  without start_cells, the volumes that fell in during the last
                  step (len(merged_into) - cells, 0 when merged_into is None)
  size_vs_start   total volume now / start_total_volume
  energy_spread   std / mean of energy per unit volume (packing)
  lumps           connected groups of neighbouring energy volumes whose packing
                  (energy / volume) is at least LUMP_FACTOR (2.0) times the mean
                  packing; a single packed volume is a lump of one.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from physics.cells.grid import Grid

LUMP_FACTOR = 2.0


def grid_stats(grid: Grid, start_total_volume: float, start_cells: int | None = None) -> dict:
    """Cells, collapsed so far, size against the start, packing spread and lump count."""
    if start_cells is None:
        start_cells = grid.cell_count if grid.merged_into is None else len(grid.merged_into)
    packing = grid.energy / grid.volume
    return {"cells": grid.cell_count,
            "collapsed": start_cells - grid.cell_count,
            "size_vs_start": float(grid.volume.sum() / start_total_volume),
            "energy_spread": float(packing.std() / packing.mean()),
            "lumps": _lumps(grid, packing >= LUMP_FACTOR * packing.mean())}


def _lumps(grid: Grid, packed: np.ndarray) -> int:
    inner = grid.joints[packed[grid.joints[:, 0]] & packed[grid.joints[:, 1]]]
    links = sp.coo_matrix((np.ones(len(inner)), (inner[:, 0], inner[:, 1])),
                          shape=(grid.cell_count, grid.cell_count))
    _, label = connected_components(links, directed=False)
    return len(np.unique(label[packed]))
