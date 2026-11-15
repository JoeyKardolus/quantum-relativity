"""Adjacency: a graph's touching pairs as one symmetric sparse matrix.

Contract: `adjacency(low, high, node_count)` takes two int arrays of equal length, one
entry per touching pair (node ids 0 <= id < node_count, low != high), and returns a
scipy CSR matrix of shape (node_count, node_count) holding 1 for every pair, both ways.
A pair listed twice counts once. The graph is unweighted: only which nodes touch, not
how much (bible rule 1: distance and dimension are measured from neighbour relations
alone). Raises ValueError for unequal lengths, an id out of range or a self pair.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def adjacency(low: np.ndarray, high: np.ndarray, node_count: int) -> sparse.csr_matrix:
    """The symmetric 0/1 adjacency matrix of the listed pairs."""
    low = np.asarray(low, dtype=np.int64)
    high = np.asarray(high, dtype=np.int64)
    if low.shape != high.shape or low.ndim != 1:
        raise ValueError(f"low and high must be equal-length 1D arrays, got {low.shape} and {high.shape}")
    if low.size and (min(low.min(), high.min()) < 0 or max(low.max(), high.max()) >= node_count):
        raise ValueError(f"node ids must lie in [0, {node_count})")
    if np.any(low == high):
        raise ValueError("a node cannot touch itself")
    rows = np.concatenate([low, high])
    columns = np.concatenate([high, low])
    matrix = sparse.csr_matrix((np.ones(rows.size), (rows, columns)), shape=(node_count, node_count))
    matrix.data[:] = 1.0
    return matrix
