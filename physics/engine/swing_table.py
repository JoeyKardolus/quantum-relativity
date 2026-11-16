"""Swing table: for the joint swing, each energy volume's swinging joints in one fixed order, on the GPU.

Contract: `swing_table(state)` returns a dict the joint swing reads (joint_swing.py):
  joints        bool (universes, capacity): the joints that swing, kind -1 holding quanta
                (state.wanted > 0), in a universe with at least 2 of them
  paired        bool, same shape: joints lying between two energy volumes
  between       Warp vec2i (universes, capacity): state.between
  offsets       Warp int32 (universes, capacity + 1) and members Warp int32 (universes,
                most): volume v's swinging joints are members[u, offsets[u, v] :
                offsets[u, v + 1]], in rising joint id (a joint is listed once under each
                of its volumes), so a sum over a volume's joints always runs in the
                same order (one seed, one run)
  members_host, volumes_host  the same list on the host with each entry's volume, and
                used, how many entries each universe has
The table follows the cell graph (state.kind, state.between) and which joints hold
quanta; it is kept on the state and built again only when one of them changed (a
collapse or a split), so a step without either costs one comparison.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from physics.engine.state import State


def swing_table(state: State) -> dict:
    """Each volume's swinging joints in rising id, built again only when the cell graph changed."""
    joints = (state.kind == -1) & (state.wanted > 0.0)
    joints &= joints.sum(axis=1, keepdims=True) >= 2
    kept = state.swing
    if (kept is not None and kept["joints"].shape == joints.shape and np.array_equal(kept["joints"], joints)
            and np.array_equal(kept["sides"], state.between)):
        return kept
    device = state.quanta.device
    universes, capacity = joints.shape
    lists = [_members(joints[universe], state.between[universe]) for universe in range(universes)]
    most = max(1, max(members.size for members, _ in lists))
    members_host = np.zeros((universes, most), dtype=np.int32)
    volumes_host = np.zeros((universes, most), dtype=np.int32)
    offsets = np.zeros((universes, capacity + 1), dtype=np.int32)
    used = np.zeros(universes, dtype=np.int64)
    for universe, (members, volumes) in enumerate(lists):
        used[universe] = members.size
        members_host[universe, :members.size] = members
        volumes_host[universe, :members.size] = volumes
        offsets[universe, 1:] = np.cumsum(np.bincount(volumes, minlength=capacity))
    state.swing = {
        "joints": joints, "sides": state.between.copy(), "paired": (state.between >= 0).all(axis=-1),
        "between": wp.array(np.ascontiguousarray(state.between.astype(np.int32)), dtype=wp.vec2i, device=device),
        "offsets": wp.array(offsets, dtype=wp.int32, device=device),
        "members": wp.array(members_host, dtype=wp.int32, device=device),
        "members_host": members_host, "volumes_host": volumes_host, "used": used}
    return state.swing


def _members(joints: np.ndarray, between: np.ndarray) -> tuple:
    """One universe's (joint, volume) entries sorted by volume, then joint id."""
    ids = np.flatnonzero(joints)
    member = np.concatenate([ids, ids])
    volume = np.concatenate([between[ids, 0], between[ids, 1]])
    keep = volume >= 0
    member, volume = member[keep], volume[keep]
    order = np.lexsort((member, volume))
    return member[order].astype(np.int32), volume[order].astype(np.int32)
