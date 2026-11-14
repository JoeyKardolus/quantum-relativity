"""Lumps: which energy volumes form lumps, and which lumps have sealed.

Contract: `lumps(grid)` returns (label, sealed):
  label   per energy volume its lump number 0.., numbered in order of lowest volume,
          or -1 when it is not packed. A lump is a connected group of neighbouring
          energy volumes whose packing (energy / volume) is at least LUMP_FACTOR (2.0)
          times the mean packing; one packed volume alone is a lump of one.
  sealed  per lump, True when every outer seam (a seam around at least one of its
          volumes and at least one volume outside it) is closed, seam_open 0; a lump
          with no outer seam counts as sealed.
Reads the Grid only: energies, sizes, joints and seams.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from physics.cells.grid import Grid

LUMP_FACTOR = 2.0


def lumps(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Lump number per energy volume (-1 outside any lump) and whether each lump has sealed."""
    packing = grid.energy / grid.volume
    label = _lump_labels(grid, packing >= LUMP_FACTOR * packing.mean())
    sealed = np.ones(int(label.max()) + 1, dtype=bool)
    sealed[_leaky_lumps(grid, label)] = False
    return label, sealed


def _lump_labels(grid: Grid, packed: np.ndarray) -> np.ndarray:
    """Per energy volume its lump, numbered 0.. in order of lowest volume; -1 if not packed."""
    inner = grid.joints[packed[grid.joints[:, 0]] & packed[grid.joints[:, 1]]]
    links = sp.coo_matrix((np.ones(len(inner)), (inner[:, 0], inner[:, 1])),
                          shape=(grid.cell_count, grid.cell_count))
    _, component = connected_components(links, directed=False)
    label = np.full(grid.cell_count, -1, dtype=np.int64)
    _, label[packed] = np.unique(component[packed], return_inverse=True)
    return label


def _leaky_lumps(grid: Grid, label: np.ndarray) -> np.ndarray:
    """Numbers of the lumps with at least one open outer seam."""
    around = label[grid.seams[grid.seam_open > 0]]
    whole = (around == around[:, :1]).all(axis=1, keepdims=True)
    return np.unique(around[(around >= 0) & ~whole])
