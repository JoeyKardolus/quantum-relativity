"""Joint collapse: joints that gave way let their two energy volumes fall into each other.

Contract: `joint_collapse(grid, giving_way)` takes a boolean per joint and returns
a new Grid (bible rule 8). Only the joints marked giving way join energy volumes;
apply_rules() marks exactly those whose volume fell below push_limit times their rest volume.
  merge     the volumes on both sides of every joint that gives way become one
            energy volume; chains of collapses in one step become one volume
            (connected components of the giving-way joints, i.e. union-find).
            New volumes are numbered in order of their lowest old volume.
  energy    energies of merged volumes add: total_energy() is kept exactly.
  charge    charges of merged volumes add.
  volume    rest_volume of the summed energy: the merged volume is more packed.
  joints    every joint is relabelled to the new volumes; joints inside a merged
            volume are dropped, joints that now separate the same two volumes
            combine into one: their volumes add, their squeeze is the largest.
  space     a merged volume's freed space (the old volumes' sum minus its rest
            volume, plus the volume of every joint that ended up inside it) goes
            to the joints around it, in proportion to their volumes
            (share_space.py), so they swell: total_space() is kept. Only when no
            joint is left around a merged volume (everything fell into one) is
            its freed space lost.
  merged_into[old_volume] = its new index (identity when nothing gave way).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from physics.cells.grid import Grid
from physics.cells.rest_volume import rest_volume
from physics.cells.share_space import share_space


def joint_collapse(grid: Grid, giving_way: np.ndarray) -> Grid:
    """Contract every joint that gave way, keeping all energy and all space."""
    if not giving_way.any():
        return replace(grid, merged_into=np.arange(grid.cell_count))
    merged_into = _merge_map(grid, giving_way)
    count = int(merged_into.max()) + 1
    energy = np.bincount(merged_into, grid.energy, count)
    volume = rest_volume(energy)
    ends = np.sort(merged_into[grid.joints], axis=1)
    inside = ends[:, 0] == ends[:, 1]
    freed = (np.bincount(merged_into, grid.volume, count) - volume
             + np.bincount(ends[inside, 0], grid.joint_volume[inside], count))
    joints, slot = np.unique(ends[~inside], axis=0, return_inverse=True)
    joints = joints.astype(np.int64).reshape(-1, 2)
    slot = slot.ravel()
    squeeze = np.zeros(len(joints))
    np.maximum.at(squeeze, slot, grid.squeeze[~inside])
    joint_volume = np.bincount(slot, grid.joint_volume[~inside], len(joints))
    return Grid(cell_count=count, joints=joints, energy=energy, volume=volume,
                joint_volume=share_space(joints, joint_volume, freed), squeeze=squeeze,
                charge=np.bincount(merged_into, grid.charge, count), merged_into=merged_into)


def _merge_map(grid: Grid, giving_way: np.ndarray) -> np.ndarray:
    pairs = grid.joints[giving_way]
    links = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                          shape=(grid.cell_count, grid.cell_count))
    _, label = connected_components(links, directed=False)
    _, first = np.unique(label, return_index=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first)] = np.arange(len(first))
    return rank[label]
