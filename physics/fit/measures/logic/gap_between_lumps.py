"""Gap between lumps: the distance between the lumps, counted through the cells between them.

Contract: `gap_between_lumps(snapshots)` takes `take_snapshot` dicts in time order and
returns one number per snapshot, or None for all when the run ends with fewer than two
lumps (or with no chain between any two).
A lump is a group of lump volumes joined through ways in the last snapshot; its centre
is its volume holding the most quanta; the CENTRES lumps holding the most are followed.
The centres are cells, the same cells in every snapshot (a centre that is a tear's new
cell, with no row in the first snapshot, is left out). The cells between two centres are
fixed once, in the first snapshot: the chain volume, joint, volume, ... along ways with
the fewest cells. No length is read to choose it, or a route picked for its short, full
volumes would lengthen by itself as energy evens out. Where several joints carry a way
between the same two volumes, the chain takes the way in the lowest row. The distance is
the summed lengths of those same cells in each snapshot (the two end cells are not
counted): space between lumps as the bible measures it, a sum of cell sizes. The number
for a snapshot is the mean over all pairs.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from physics.fit.measures.definitions.lump_groups import lump_groups
from physics.fit.measures.definitions.lump_marks import CENTRES


def _centres(last: dict, first_count: int) -> np.ndarray:
    """The fullest volume of each of the fullest lumps of the last snapshot, as ids that existed at the start."""
    lumps, held = last["lumps"], last["held"]
    if len(lumps) < 2:
        return np.zeros(0, dtype=np.int64)
    packed = np.zeros(last["count"], dtype=bool)
    packed[lumps] = True
    labels = lump_groups(last["count"], packed, last["ways"][1], last["ways"][2])[0][lumps]
    order = np.lexsort((-held[lumps], labels))
    fullest = lumps[order][np.unique(labels[order], return_index=True)[1]]
    amounts = np.bincount(labels, weights=held[lumps])
    chosen = fullest[np.argsort(-amounts, kind="stable")[:CENTRES]]
    return np.sort(chosen[chosen < first_count])


def _carriers(first: dict) -> tuple:
    """Per pair of volumes with a way between them, the joint of the lowest such row; and the pairs as a graph."""
    joints, near, far = first["ways"]
    low, high = np.minimum(near, far), np.maximum(near, far)
    apart = low != high
    carrier: dict = {}
    for joint, one, other in zip(joints[apart].tolist(), low[apart].tolist(), high[apart].tolist()):
        carrier.setdefault((one, other), joint)
    graph = coo_matrix((np.ones(int(apart.sum()), dtype=np.int8), (low[apart], high[apart])),
                       shape=(first["count"], first["count"])).tocsr()
    return carrier, graph


def _chain(carrier: dict, before: np.ndarray, start: int, end: int) -> np.ndarray | None:
    """The cells between two centres along the found route, or None when no route joins them."""
    cells, here = [], end
    while here != start and before[here] >= 0:
        back = int(before[here])
        cells += [carrier[(min(back, here), max(back, here))], back]
        here = back
    return np.array(cells[:-1], dtype=np.int64) if here == start else None


def _chains(first: dict, centres: np.ndarray) -> list:
    """Per pair of centres, the ids of the cells between them along the chain of ways with the fewest cells."""
    carrier, graph = _carriers(first)
    before = dijkstra(graph, directed=False, unweighted=True, indices=centres, return_predecessors=True)[1]
    chains = []
    for row, start in enumerate(centres.tolist()):
        for end in centres[row + 1:].tolist():
            chain = _chain(carrier, before[row], start, end)
            if chain is not None:
                chains.append(chain)
    return chains


def gap_between_lumps(snapshots: list) -> list:
    """The mean distance between the end's lumps at every snapshot, None for all without two lumps."""
    centres = _centres(snapshots[-1], snapshots[0]["count"])
    chains = _chains(snapshots[0], centres) if len(centres) >= 2 else []
    if not chains:
        return [None] * len(snapshots)
    return [float(np.mean([shot["length"][chain].sum() for chain in chains])) for shot in snapshots]
