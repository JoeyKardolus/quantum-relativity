"""Largest group: the size of the biggest group of members joined to each other through pairs.

Contract: `largest_group(member, first, second)` takes a bool array, one entry per node,
and two equal-length int arrays naming pairs of nodes. Two members are joined when a
pair names both; a group is every member reachable that way. Returns the number of
members in the biggest group as an int: 0 with no member, 1 when no two members are
joined. A pair with an end that is no member joins nothing. Raises ValueError for pair
arrays of unequal length.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def largest_group(member: np.ndarray, first: np.ndarray, second: np.ndarray) -> int:
    """How many members the biggest joined group holds."""
    member = np.asarray(member, dtype=bool)
    first, second = np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)
    if first.shape != second.shape:
        raise ValueError(f"pairs need two equal lists, got {first.shape} and {second.shape}")
    if not member.any():
        return 0
    both = member[first] & member[second]
    joined = coo_matrix((np.ones(int(both.sum())), (first[both], second[both])), shape=(member.size, member.size))
    _, group = connected_components(joined, directed=False)
    return int(np.bincount(group[member]).max())
