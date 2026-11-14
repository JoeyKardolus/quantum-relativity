"""Joint collapse: joints that gave way let their two energy volumes fall into each other.

Contract: `joint_collapse(grid, giving_way)` takes a boolean per joint and returns
a new Grid (bible rule 8):
  merge     the volumes on both sides of every joint that gives way become one
            energy volume; chains of collapses in one step become one volume
            (connected components of the giving-way joints, i.e. union-find).
            New volumes are numbered in order of their lowest old volume.
  energy    energies of merged volumes add: total_energy() is kept exactly.
  volume    rest_volume of the summed energy: the merged volume is more packed.
  joints    every joint is relabelled to the new volumes; joints inside a merged
            volume are dropped, joints that now separate the same two volumes
            combine into one, keeping the largest squeeze.
  merged_into[old_volume] = its new index (identity when nothing gave way).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from physics.cells.grid import Grid
from physics.cells.rest_volume import rest_volume


def joint_collapse(grid: Grid, giving_way: np.ndarray) -> Grid:
    """Contract every joint that gave way, keeping all energy."""
    if not giving_way.any():
        return replace(grid, merged_into=np.arange(grid.cell_count))
    merged_into = _merge_map(grid, giving_way)
    count = int(merged_into.max()) + 1
    energy = np.bincount(merged_into, grid.energy, count)
    ends = np.sort(merged_into[grid.joints], axis=1)
    inside = ends[:, 0] == ends[:, 1]
    joints, slot = np.unique(ends[~inside], axis=0, return_inverse=True)
    squeeze = np.zeros(len(joints))
    np.maximum.at(squeeze, slot.ravel(), grid.squeeze[~inside])
    return Grid(cell_count=count, joints=joints.astype(np.int64).reshape(-1, 2), energy=energy,
                volume=rest_volume(energy), squeeze=squeeze, merged_into=merged_into)


def _merge_map(grid: Grid, giving_way: np.ndarray) -> np.ndarray:
    pairs = grid.joints[giving_way]
    links = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                          shape=(grid.cell_count, grid.cell_count))
    _, label = connected_components(links, directed=False)
    _, first = np.unique(label, return_index=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first)] = np.arange(len(first))
    return rank[label]
