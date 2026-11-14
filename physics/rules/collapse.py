"""Joint collapse: joints that gave way fall into a neighbour, keeping all their energy.

Contract: `collapse(network, giving_way)` takes a boolean per joint and returns
a new Network (bible rule 8):
  merge      every joint that gives way (and has a connection) merges into the
             joint at the other end of its shortest connection; chains and
             mutual merges become one joint (connected components of the merge
             links). New joints are numbered in order of their lowest old joint.
  energy     a connection between two joints that merged together would join a
             joint to itself; its energy is shared equally among the merged
             joint's remaining connections. Connections that now join the same
             two joints combine into one, energies added. A merged joint left
             with no connection (a whole isolated cluster fell together) keeps
             that energy as an isolated joint: Network.held. Energy already held
             moves with its joint. total_energy() is kept exactly; never raises.
  length     every connection whose energy changed re-follows the energy-length
             rule (length = LENGTH_SCALE / energy); others keep their length.
  joints     charges of merged joints add; the causal order, when present, is
             carried over (merged precedes merged when any old pair did).
  merged_into[old_joint] = new index of that joint (identity when none gave way).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from physics.network import Network, energy_length


def collapse(network: Network, giving_way: np.ndarray) -> Network:
    """Merge every joint that gave way into the far end of its shortest connection."""
    if not giving_way.any():
        return replace(network, merged_into=np.arange(network.joint_count))
    merged_into = _merge_map(network, giving_way)
    joint_count = int(merged_into.max(initial=-1)) + 1
    ends = merged_into[network.pairs]
    looped = ends[:, 0] == ends[:, 1]
    share, stranded = _loop_energy_share(ends, looped, network.energy, joint_count)
    share = share[~looped]
    energy = network.energy[~looped] + share
    length = np.where(share > 0, energy_length(energy), network.length[~looped])
    pairs, energy, length = _combine(np.sort(ends[~looped], axis=1), energy, length)
    charge = np.bincount(merged_into, network.charge, joint_count)
    return replace(network, joint_count=joint_count, pairs=pairs, energy=energy,
                   length=length, charge=charge, causal=_causal(network, merged_into, joint_count),
                   merged_into=merged_into,
                   held=_held(network, merged_into, joint_count, stranded))


def _merge_map(network: Network, giving_way: np.ndarray) -> np.ndarray:
    joints = network.pairs.ravel(order="F")
    others = network.pairs[:, ::-1].ravel(order="F")
    order = np.lexsort((np.tile(network.length, 2), joints))
    holders, shortest = np.unique(joints[order], return_index=True)
    nearest = np.full(network.joint_count, -1)
    nearest[holders] = others[order][shortest]
    movers = np.flatnonzero(giving_way & (nearest >= 0))
    links = sp.coo_matrix((np.ones(len(movers)), (movers, nearest[movers])),
                          shape=(network.joint_count, network.joint_count))
    _, label = connected_components(links, directed=False)
    _, first = np.unique(label, return_index=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first)] = np.arange(len(first))
    return rank[label]


def _loop_energy_share(ends: np.ndarray, looped: np.ndarray, energy: np.ndarray,
                       joint_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Energy each surviving connection gains from the loops at its two joints, and the
    loop energy of each new joint left with no connection to give it to."""
    loop_energy = np.bincount(ends[looped, 0], energy[looped], joint_count)
    kept = ends[~looped]
    holders = np.bincount(kept.ravel(), minlength=joint_count)
    stranded = np.where(holders == 0, loop_energy, 0.0)
    per_holder = np.divide(loop_energy, holders, out=np.zeros(joint_count), where=holders > 0)
    share = np.zeros(len(ends))
    share[~looped] = per_holder[kept[:, 0]] + per_holder[kept[:, 1]]
    return share, stranded


def _held(network: Network, merged_into: np.ndarray, joint_count: int, stranded: np.ndarray):
    """Energy each new isolated joint holds: what its old joints held plus stranded loops."""
    before = np.zeros(network.joint_count) if network.held is None else network.held
    held = np.bincount(merged_into, before, joint_count) + stranded
    return held if held.any() else None


def _combine(pairs: np.ndarray, energy: np.ndarray, length: np.ndarray):
    """One connection per joint pair; duplicates add their energy and re-follow length."""
    unique, slot, count = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    slot = slot.ravel()
    total = np.bincount(slot, energy, len(unique))
    kept = np.zeros(len(unique))
    kept[slot] = length
    return unique.astype(np.int64), total, np.where(count > 1, energy_length(total), kept)


def _causal(network: Network, merged_into: np.ndarray, joint_count: int):
    if network.causal is None:
        return None
    rows = np.arange(network.joint_count)
    fold = sp.csr_matrix((np.ones(network.joint_count), (rows, merged_into)),
                         shape=(network.joint_count, joint_count))
    relation = (fold.T @ network.causal.astype(np.int64) @ fold).tocsr()
    relation.setdiag(0)
    relation.eliminate_zeros()
    return relation > 0
