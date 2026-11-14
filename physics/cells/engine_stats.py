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
  sealed_lumps   lumps whose outer seams (seams around at least one of its volumes
                 and at least one volume outside it) are all closed: seam_open 0
                 (a lump with no outer seam counts as sealed)
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
    """Cells, joints, collapses so far, size against the start, packing spread, lumps, sealed lumps."""
    grid = engine.grid
    packing = grid.energy / grid.volume
    label = _lump_labels(grid, packing >= LUMP_FACTOR * packing.mean())
    lumps = int(label.max()) + 1
    return {"cells": grid.cell_count,
            "joints": len(grid.joints),
            "collapses": engine.collapses,
            "size_vs_start": float(grid.total_space() / engine.start_space),
            "energy_spread": float(packing.std() / packing.mean()),
            "lumps": lumps,
            "sealed_lumps": lumps - _leaky_lumps(grid, label)}


def _lump_labels(grid: Grid, packed: np.ndarray) -> np.ndarray:
    """Per energy volume its lump, numbered 0.. in order of lowest volume; -1 if not packed."""
    inner = grid.joints[packed[grid.joints[:, 0]] & packed[grid.joints[:, 1]]]
    links = sp.coo_matrix((np.ones(len(inner)), (inner[:, 0], inner[:, 1])),
                          shape=(grid.cell_count, grid.cell_count))
    _, component = connected_components(links, directed=False)
    label = np.full(grid.cell_count, -1, dtype=np.int64)
    _, label[packed] = np.unique(component[packed], return_inverse=True)
    return label


def _leaky_lumps(grid: Grid, label: np.ndarray) -> int:
    """Number of lumps with at least one open outer seam."""
    around = label[grid.seams[grid.seam_open > 0]]
    whole = (around == around[:, :1]).all(axis=1, keepdims=True)
    return len(np.unique(around[(around >= 0) & ~whole]))
