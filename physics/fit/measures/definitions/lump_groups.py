"""Lump groups: lump volumes joined through ways are one lump.

Contract: `lump_groups(count, packed, first, second)` returns `(numbered, lumps)`.
`packed` says per cell id (of `count`) whether it is a lump volume; `first` and `second`
are the two energy volumes of every way. Two lump volumes with a way between them belong
to the same lump, and so on along ways: a lump is a region, not a single cell.
`numbered[cell]` is the lump's number, 0 .. lumps - 1; only "same number, same lump" is
promised. It is 0 and meaningless for a cell that is no lump volume. `lumps` is how many
lumps there are.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def lump_groups(count: int, packed: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple:
    """Per cell the number of its lump, and how many lumps there are."""
    numbered = np.zeros(count, dtype=np.int64)
    if not packed.any():
        return numbered, 0
    joined = packed[first] & packed[second]
    links = coo_matrix((np.ones(int(joined.sum()), dtype=np.int8), (first[joined], second[joined])),
                       shape=(count, count))
    labels = connected_components(links, directed=False)[1]
    numbered[packed] = np.unique(labels[packed], return_inverse=True)[1]
    return numbered, int(numbered[packed].max()) + 1
