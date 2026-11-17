"""fit_gap: the distance between the lumps, counted through the cells between them, at every snapshot of a run.

Contract: fit_gap(snapshots, fell_into) -> (gaps, held_share).
`snapshots` are fit_snapshot dicts in time order; `fell_into` is the State's record
{vanished volume: the volume it fell into}.
  gaps        one number per snapshot, or None for all when the run ends with fewer than
              two lumps. A lump is a group of lump volumes joined through ways in the
              last snapshot; its centre is its volume holding the most quanta; the
              CENTRES (12) lumps holding the most are followed. The centres are cells,
              the same cells in every snapshot (a centre that is a tear's new cell,
              with no row in the first snapshot, is left out). The cells between two
              centres are fixed once, in the first snapshot: the chain volume, joint,
              volume, ... along ways with the fewest cells (the straight way; no
              length is read to choose it, or a route picked for its short, full
              volumes would lengthen by itself as energy evens out). The distance is
              the summed lengths of those same cells in each snapshot (the two end
              cells are not counted; a cell that is gone counts 0): space between
              lumps as the bible measures it, a sum of cell sizes. gaps[i] is the mean
              over all pairs in snapshot i.
  held_share  lumps that hold: of the plus quanta at the snapshot two thirds through,
              the share that sat in lump volumes then whose cell (or the cell it has
              since fallen into) is a lump volume in the last snapshot. 0 with no lumps.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

CENTRES = 12


def fit_gap(snapshots: list, fell_into: dict) -> tuple:
    """The mean distance between the end's lumps at every snapshot, and the share of energy in lumps that held."""
    centres = _centres(snapshots[-1], snapshots[0]["count"])
    chains = _chains(snapshots[0], centres) if len(centres) >= 2 else []
    gaps = [float(np.mean([_summed(shot, chain) for chain in chains])) if chains else None for shot in snapshots]
    return gaps, _held_share(snapshots[(2 * (len(snapshots) - 1)) // 3], snapshots[-1], fell_into)


def _centres(last: dict, first_count: int) -> np.ndarray:
    """The fullest volume of each of the fullest lumps of the last snapshot, as cell ids that existed at the start."""
    lumps, held = last["lumps"], last["held"]
    if len(lumps) < 2:
        return np.zeros(0, dtype=np.int64)
    _, first, second = last["ways"]
    packed = np.zeros(last["count"], dtype=bool)
    packed[lumps] = True
    joined = packed[first] & packed[second]
    links = coo_matrix((np.ones(int(joined.sum()), dtype=np.int8), (first[joined], second[joined])),
                       shape=(last["count"], last["count"]))
    labels = connected_components(links, directed=False)[1][lumps]
    order = np.lexsort((-held[lumps], labels))
    fullest = lumps[order][np.unique(labels[order], return_index=True)[1]]
    amounts = np.bincount(np.unique(labels, return_inverse=True)[1], weights=held[lumps])
    chosen = fullest[np.argsort(-amounts, kind="stable")[:CENTRES]]
    return np.sort(chosen[chosen < first_count])


def _chains(first: dict, centres: np.ndarray) -> list:
    """Per pair of centres, the ids of the cells between them along the chain of ways with the fewest cells."""
    joints, near, far = first["ways"]
    low, high = np.minimum(near, far), np.maximum(near, far)
    apart = low != high
    carrier = {}
    for joint, one, other in zip(joints[apart].tolist(), low[apart].tolist(), high[apart].tolist()):
        carrier.setdefault((one, other), joint)
    graph = coo_matrix((np.ones(int(apart.sum()), dtype=np.int8), (low[apart], high[apart])),
                       shape=(first["count"], first["count"])).tocsr()
    _, before = dijkstra(graph, directed=False, unweighted=True, indices=centres, return_predecessors=True)
    chains = []
    for row, start in enumerate(centres.tolist()):
        for end in centres[row + 1:].tolist():
            cells, here = [], end
            while here != start and before[row, here] >= 0:
                step = int(before[row, here])
                cells += [carrier[(min(step, here), max(step, here))], step]
                here = step
            if here == start:
                chains.append(np.array(cells[:-1], dtype=np.int64))
    return chains


def _summed(shot: dict, chain: np.ndarray) -> float:
    """The summed lengths, in one snapshot, of a chain's cells (a cell that is gone has length 0)."""
    return float(shot["length"][chain].astype(np.float64).sum())


def _held_share(then: dict, last: dict, fell_into: dict) -> float:
    """Share of the plus quanta of `then` in lump volumes whose cell, or the cell it fell into, is a lump volume at the end."""
    if len(then["lumps"]) == 0 or len(last["lumps"]) == 0:
        return 0.0
    still = set(last["lumps"].tolist())
    kept = 0
    for cell in then["lumps"].tolist():
        now = cell
        while now in fell_into:
            now = fell_into[now]
        if now in still:
            kept += int(then["held"][cell])
    return float(kept / then["plus"])
