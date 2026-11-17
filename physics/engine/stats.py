"""Stats: what the universe holds now, beside where it started (host numbers).

Contract: `stats(state, settings=None)` returns a dict of plain Python numbers:
  step, cells, volumes, joints   the counter and the living cells by kind
  amount, plus, minus            the exact quanta (totals.py); made, taken_joint,
                    taken_volumes, torn, crossed since the start
  books_gap         amount - (start amount + made - taken_joint - taken_volumes): 0
                    while the books are exact (axiom 1)
  collapses, merges, tears   joints collapsed, volumes that fell into another,
                    volumes torn
  size              the size of the universe: the sum of every living cell's length;
                    volume_size and joint_size its two parts; size_growth size over the
                    start's (state.start["size"])
  lumps             energy volumes holding at least LUMP_FACTOR (2) times the start's
                    mean volume quanta (at most half its length); lump_regions the
                    groups of lumps joined through ways; largest_region the most
                    volumes in one group
  squeezed, sealed  joints with squeeze above 1 now; joints settings.seal_thin (the
                    default Settings when none is given) or more times thinner than
                    the start's mean joint; squeeze_max
  clock_min, clock_mean   clock rates over the living cells
  length_min, length_max  over the energy volumes
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from physics.engine.settings import Settings
from physics.engine.state import ENERGY_VOLUME, JOINT, State
from physics.engine.totals import totals
from physics.engine.way_list import way_list

LUMP_FACTOR = 2.0


def stats(state: State, settings: Settings | None = None) -> dict:
    """Count the universe now: cells, books, events, size, lumps, squeeze and clocks."""
    kind = state.kind_h[:state.count]
    held, length, rate = state.host("cell_quanta"), state.host("length"), state.host("rate")
    squeeze = state.host("squeeze")
    volumes, joints = kind == ENERGY_VOLUME, kind == JOINT
    found = totals(state)
    start = state.start or found
    seal = (settings or Settings()).seal_thin
    size = float(length[kind != 0].astype(np.float64).sum())
    lumps = volumes & (held >= LUMP_FACTOR * state.start_quanta[ENERGY_VOLUME])
    regions, largest = _regions(state, lumps)
    found.update(
        step=state.step, cells=int((kind != 0).sum()), volumes=int(volumes.sum()), joints=int(joints.sum()),
        books_gap=found["amount"] - (start["amount"] + found["made"] - found["taken_joint"] - found["taken_volumes"]),
        collapses=state.collapses, merges=state.merges, tears=state.tears,
        size=size, volume_size=float(length[volumes].astype(np.float64).sum()),
        joint_size=float(length[joints].astype(np.float64).sum()), size_growth=size / start.get("size", size),
        lumps=int(lumps.sum()), lump_regions=regions, largest_region=largest,
        squeezed=int((squeeze[joints] > 1.0).sum()),
        sealed=int((held[joints] >= seal * state.start_quanta[JOINT]).sum()),
        squeeze_max=float(squeeze[joints].max()), clock_min=float(rate[kind != 0].min()),
        clock_mean=float(rate[kind != 0].mean()), length_min=float(length[volumes].min()),
        length_max=float(length[volumes].max()))
    return found


def _regions(state: State, lumps: np.ndarray) -> tuple:
    """Groups of lump volumes joined through ways: (how many groups, volumes in the largest)."""
    if not lumps.any():
        return 0, 0
    _, first, second = way_list(state)
    joined = lumps[first] & lumps[second]
    links = coo_matrix((np.ones(int(joined.sum()), dtype=np.int8), (first[joined], second[joined])),
                       shape=(state.count, state.count))
    _, labels = connected_components(links, directed=False)
    sizes = np.bincount(labels[lumps])
    return int((sizes > 0).sum()), int(sizes.max())
