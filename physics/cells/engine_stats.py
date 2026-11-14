"""Engine stats: what the grid looks like now, read from energies, sizes and the neighbour graph only.

Contract: `engine_stats(engine)` returns a dict:
  cells          energy volumes now
  joints         joints now (pairs of neighbouring energy volumes)
  collapses      energy volumes that fell into others since the start
  size_vs_start  total_space() now / at the start (energy volumes plus joints)
  energy_spread  std / mean of energy per unit volume (packing)
  lumps          connected groups of neighbouring energy volumes whose packing is at
                 least LUMP_FACTOR (2.0) times the mean packing; one packed volume
                 alone is a lump of one
No position, weight or shape is read.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from physics.cells.engine import Engine
from physics.cells.grid import Grid

LUMP_FACTOR = 2.0


def engine_stats(engine: Engine) -> dict:
    """Cells, joints, collapses so far, size against the start, packing spread, lumps."""
    grid = engine.grid
    packing = grid.energy / grid.volume
    return {"cells": grid.cell_count,
            "joints": len(grid.joints),
            "collapses": engine.collapses,
            "size_vs_start": float(grid.total_space() / engine.start_space),
            "energy_spread": float(packing.std() / packing.mean()),
            "lumps": _lumps(grid, packing >= LUMP_FACTOR * packing.mean())}


def _lumps(grid: Grid, packed: np.ndarray) -> int:
    inner = grid.joints[packed[grid.joints[:, 0]] & packed[grid.joints[:, 1]]]
    links = sp.coo_matrix((np.ones(len(inner)), (inner[:, 0], inner[:, 1])),
                          shape=(grid.cell_count, grid.cell_count))
    _, label = connected_components(links, directed=False)
    return len(np.unique(label[packed]))
