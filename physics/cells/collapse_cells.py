"""Collapse cells: the mesh after joints give way; their two energy volumes become one cell.

Contract: `collapse_cells(faces, face_cells, face_joint, joints, giving_way, cell_count)`
returns (faces, face_cells, face_origin, merged_into). giving_way is a boolean per
joint (step.py marks those below push_limit times their rest thickness; bible
rule 8). The cells on both sides of every joint giving way become one cell; chains
in one step become one (connected components of the giving-way joints). New cells
are numbered in order of their lowest old cell; merged_into[old] = new.
Every face between two cells that become one is removed (the joint is gone, the
two fall into each other); every other face is kept with its cells relabelled and
its ring and orientation unchanged, so the merged cell is the union of its old
cells' faces. face_origin[f] is the old joint of each kept face (-1 outside), so a
neighbour's two joints with the two merged cells become one (face_joints.py).
Corners are not moved; a corner no kept face uses is left for drop_loose_corners().
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def collapse_cells(faces: Sequence[np.ndarray], face_cells: np.ndarray, face_joint: np.ndarray,
                   joints: np.ndarray, giving_way: np.ndarray, cell_count: int):
    """Faces, their cells and old joints after the cells of every giving-way joint become one."""
    pairs = joints[giving_way]
    links = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                          shape=(cell_count, cell_count))
    _, label = connected_components(links, directed=False)
    _, first = np.unique(label, return_index=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first)] = np.arange(len(first))
    merged_into = rank[label]
    relabelled = np.where(face_cells >= 0, merged_into[np.maximum(face_cells, 0)], -1)
    kept = np.flatnonzero(relabelled[:, 0] != relabelled[:, 1])
    return ([faces[index] for index in kept], relabelled[kept], face_joint[kept].copy(),
            merged_into)
