"""Group label: the connected groups of marked boxes in one universe, joined round its edges when it wraps, on the host.

Contract: `group_label(members, wrap, corners=False)` takes a bool array (side, side,
side) and returns (labels, count): an int array numbering each face-connected group of
True boxes from 1 (26-connected with corners=True), 0 elsewhere, and how many groups.
When the universe wraps (state.wrap) groups that meet across opposite faces are one.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def group_label(members: np.ndarray, wrap: bool, corners: bool = False) -> tuple:
    """Connected groups of `members` (labels from 1, and their number), joined across the edges when the universe wraps."""
    structure = np.ones((3, 3, 3), dtype=bool) if corners else None
    labels, count = ndimage.label(members, structure=structure)
    if not wrap or count == 0:
        return labels, count
    parent = np.arange(count + 1)

    def root(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label
    for axis in range(3):
        low = np.take(labels, 0, axis=axis)
        high = np.take(labels, -1, axis=axis)
        for shift in ([(0, 0)] if structure is None else [(da, db) for da in (-1, 0, 1) for db in (-1, 0, 1)]):
            moved = np.roll(high, shift, axis=(0, 1))
            both = (low > 0) & (moved > 0)
            for first, second in zip(low[both], moved[both]):
                parent[root(first)] = root(second)
    roots = np.array([root(label) for label in range(count + 1)])
    unique, renumber = np.unique(roots[1:], return_inverse=True)
    lookup = np.concatenate([[0], renumber + 1])
    return lookup[labels], int(unique.size)
