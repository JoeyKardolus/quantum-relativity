"""Spectral dimension: a graph's dimension from how fast a random walker stops coming home.

Contract: `spectral_dimension(matrix, sources, times)` takes a symmetric adjacency
matrix (adjacency.py), source node ids (each with at least one neighbour) and whole
step counts (at least two, all >= 1). A lazy walker (each step: stay with chance 1/2,
else go to a neighbour picked evenly) starts at each source; the exact chance of being
back home after t steps is followed by spreading the probability, not by sampling. In
d dimensions it falls as t**(-d / 2), so the dimension is -2 times the slope of
ln(return chance) against ln(t). The lazy step keeps the walk from ticking between
even and odd steps on a two-coloured graph such as a grid. Returns a dict:
  dimension          -2 * the least-squares slope over the times
  times              the times (ints); return_probability  the mean return chance at each
Choose times short enough that the walk does not reach the graph's edge.
Raises ValueError for fewer than two times, a time < 1, no source, or a source with no neighbour.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def spectral_dimension(matrix: sparse.csr_matrix, sources, times) -> dict:
    """-2 times the slope of ln(return chance) against ln(steps) of a lazy walk from the sources."""
    sources = np.atleast_1d(np.asarray(sources, dtype=np.int64))
    times = np.asarray(times, dtype=np.int64)
    if sources.size == 0 or times.size < 2 or times.min() < 1:
        raise ValueError(f"need sources and at least two times >= 1, got {sources.size} sources, times {times}")
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(degree[sources] == 0):
        raise ValueError("a source has no neighbour, so a walker there never moves")
    spread = sparse.diags(1.0 / np.maximum(degree, 1.0)) @ matrix
    walk = (0.5 * sparse.identity(matrix.shape[0]) + 0.5 * spread).T.tocsr()
    chance = np.zeros((matrix.shape[0], sources.size))
    chance[sources, np.arange(sources.size)] = 1.0
    wanted = set(times.tolist())
    returns = {}
    for taken in range(1, int(times.max()) + 1):
        chance = walk @ chance
        if taken in wanted:
            returns[taken] = float(np.mean(chance[sources, np.arange(sources.size)]))
    probability = np.array([returns[int(time)] for time in times])
    slope, _ = np.polyfit(np.log(times), np.log(probability), 1)
    return {"dimension": float(-2.0 * slope), "times": times.tolist(), "return_probability": probability.tolist()}
