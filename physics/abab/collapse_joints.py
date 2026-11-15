"""Collapse joints: joints past their limit vanish and the energy volumes around them fall into one (rule 8).

Contract: `collapse_joints(state, giving)` returns a new State; giving is a bool mask
over the volumes, True only on joints. Every giving joint vanishes, and the energy
volumes it touched fall into one energy volume; giving joints that share an energy
volume fall in together (connected groups over the giving joints' contacts), so the
order does not matter. The one energy volume of a group:
  index   the group's lowest energy volume (the others and the giving joints are
          dropped; remaining volumes keep their order)
  energy  all its energy volumes' energy plus the giving joints' energy, exactly
  size    the sum of its energy volumes' sizes, smaller by the joints' energy
          coming in (grow_by_energy.py: times (E_volumes / E_all) ** 3); the
          joints' own space goes with them, so space shrinks where it collapses
  rate    0
  contacts every other joint of every energy volume in the group, each once: a
          joint that touched two of them now touches the one, so ABAB holds
collapses grows by the number of giving joints. No giving joint: the state as it was.
Raises RuntimeError for a giving joint that touches no energy volume.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from physics.abab.grow_by_energy import grow_by_energy
from physics.abab.state import KIND_A, State


def collapse_joints(state: State, giving: np.ndarray) -> State:
    """The state after every giving joint vanished and its energy volumes fell into one."""
    if not giving.any():
        return state
    count = len(state.kind)
    group = _groups(state, giving)
    lead = _group_leads(state, group)
    if (lead[group[giving]] >= count).any():
        raise RuntimeError("joint collapse: a giving joint touches no energy volume, its energy has nowhere to go")
    volume_energy = np.bincount(group, np.where(state.kind == KIND_A, state.energy, 0.0), count)
    all_energy = np.bincount(group, np.where(giving, state.energy, 0.0), count) + volume_energy
    volume_size = np.bincount(group, np.where(state.kind == KIND_A, state.size, 0.0), count)
    is_lead = (lead[group] == np.arange(count)) & (state.kind == KIND_A)
    energy = state.energy.copy()
    size = state.size.copy()
    energy[is_lead] = all_energy[group[is_lead]]
    size[is_lead] = grow_by_energy(volume_size[group[is_lead]], volume_energy[group[is_lead]],
                                   all_energy[group[is_lead]])
    rate = np.where(is_lead, 0.0, state.rate)
    keep = ~giving & ((state.kind != KIND_A) | is_lead)
    renumber = np.cumsum(keep) - 1
    pairs = state.contacts[~giving[state.contacts[:, 1]]]
    moved = np.column_stack([renumber[lead[group[pairs[:, 0]]]], renumber[pairs[:, 1]]])
    return replace(state, kind=state.kind[keep], energy=energy[keep], size=size[keep], rate=rate[keep],
                   contacts=np.unique(moved, axis=0).reshape(-1, 2),
                   collapses=state.collapses + int(giving.sum()))


def _groups(state: State, giving: np.ndarray) -> np.ndarray:
    """Per volume its group: connected over the contacts of giving joints, every other volume alone."""
    count = len(state.kind)
    links = state.contacts[giving[state.contacts[:, 1]]]
    graph = sp.coo_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])), shape=(count, count))
    _, group = connected_components(graph, directed=False)
    return group


def _group_leads(state: State, group: np.ndarray) -> np.ndarray:
    """Per group its lowest energy volume (a group with none gets an index past the end)."""
    count = len(state.kind)
    lead = np.full(count, count, dtype=np.int64)
    volumes = np.flatnonzero(state.kind == KIND_A)
    np.minimum.at(lead, group[volumes], volumes)
    return lead
