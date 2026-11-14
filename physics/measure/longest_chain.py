"""Longest chain between two related joints of a causal network: discrete proper time.

Contract: `longest_chain(network, start, end)` returns the number of steps
(relations) in the longest chain start < ... < end of the causal order, or -1
when start does not precede end. In flat 1+1 spacetime it estimates proper
time T as L ~ sqrt(2 * density) * T (Brightwell-Gregory; approached from below
at finite size). Reads only `network.causal`.
"""
from __future__ import annotations

import numpy as np

from physics.network import Network


def longest_chain(network: Network, start: int, end: int) -> int:
    """Steps in the longest causal chain from joint start to joint end."""
    relation = network.causal
    if relation is None:
        raise ValueError("longest_chain needs a causal network")
    if not relation[start, end]:
        return -1
    after_start = relation[start].toarray().ravel()
    before_end = relation[:, end].toarray().ravel()
    between = np.flatnonzero(after_start & before_end)
    members = np.concatenate([[start], between, [end]])
    inner = relation[members][:, members].tocsc()
    order = np.argsort(np.asarray(inner.sum(axis=0)).ravel(), kind="stable")
    steps = np.full(len(members), -1, dtype=np.int64)
    steps[0] = 0
    for position in order[1:]:
        before = inner.indices[inner.indptr[position]:inner.indptr[position + 1]]
        steps[position] = steps[before].max() + 1
    return int(steps[-1])
