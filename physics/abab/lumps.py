"""Lumps: which energy volumes form lumps, and which lumps have sealed.

Contract: `lumps(state, openness)` returns (label, sealed):
  label   per volume its lump number 0.. (numbered in order of lowest volume), -1
          for a joint or an energy volume that is not packed. A lump is a connected
          group of packed energy volumes, two being neighbours when they touch the
          same joint; packed means packing (energy / size) at least LUMP_FACTOR
          (2.0, as in the removed cells engine) times the energy volumes' mean packing.
  sealed  per lump, True when every outer joint (one touching an energy volume of
          the lump and an energy volume outside it) has openness 0; a lump with no
          outer joint counts as sealed.
openness per volume (joint_openness.py; only joints read). Reads only kind, energy,
size and contacts.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from physics.abab.passages import passages
from physics.abab.state import KIND_A, State

LUMP_FACTOR = 2.0


def lumps(state: State, openness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lump number per volume (-1 outside any lump) and whether each lump has sealed."""
    is_volume = state.kind == KIND_A
    packing = state.energy / state.size
    packed = is_volume & (packing >= LUMP_FACTOR * packing[is_volume].mean())
    giver, joint, taker = passages(state.contacts)
    label = _labels(len(state.kind), packed, giver, taker)
    sealed = np.ones(int(label.max()) + 1, dtype=bool)
    leaky = (label[giver] >= 0) & (label[giver] != label[taker]) & (openness[joint] > 0)
    sealed[np.unique(label[giver[leaky]])] = False
    return label, sealed


def _labels(count: int, packed: np.ndarray, giver: np.ndarray, taker: np.ndarray) -> np.ndarray:
    """Per volume its lump, numbered 0.. in order of lowest volume; -1 if not packed."""
    inner = packed[giver] & packed[taker]
    links = sp.coo_matrix((np.ones(int(inner.sum())), (giver[inner], taker[inner])), shape=(count, count))
    _, component = connected_components(links, directed=False)
    label = np.full(count, -1, dtype=np.int64)
    _, label[packed] = np.unique(component[packed], return_inverse=True)
    return label
