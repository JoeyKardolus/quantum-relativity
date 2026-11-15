"""Split volumes: an energy volume grown too big tears in two with a new joint between them (rule 8).

Contract: `split_volumes(state, trigger)` returns a new State. Every energy volume
whose size is above `trigger` (step passes split_size * start_mean_size; size, not
density, as rule 8 says) splits once:
  joint     a new joint between the two halves at rest, rate 0, appended after
            every existing volume: its size JOINT_SIZE (0.03, the start's
            joint-to-volume size ratio) times the parent's size, its energy the
            energy that wants that size (joint_energy.py), both taken from the parent
  halves    the parent keeps its index as the first half, the second is appended;
            each gets half of what is left of the parent's energy and size once the
            joint took its part (the second the exact remainder, so energy and the
            total size are kept exactly)
  joints    the parent's other joints, ordered by their own energy (lowest first,
            ties by index), go alternately to the first half and the second; no
            position is read. Both halves touch the new joint, so ABAB holds
splits grows by the number of energy volumes split. A child above the trigger
splits on a later step. A volume whose energy is not above the new joint's (too
dilute to pay for a joint at rest, joint_energy() below) does not split and stays
too big; stats() counts those.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.abab.joint_energy import joint_energy
from physics.abab.joint_rest_size import JOINT_SIZE
from physics.abab.state import KIND_A, KIND_B, State


def split_volumes(state: State, trigger: float) -> State:
    """The state after every energy volume larger than trigger split in two around a new joint."""
    paid = state.energy > joint_energy(JOINT_SIZE * state.size)
    big = np.flatnonzero((state.kind == KIND_A) & (state.size > trigger) & paid)
    if len(big) == 0:
        return state
    kind, energy, size = list(state.kind), list(state.energy), list(state.size)
    rate = list(state.rate)
    contacts = state.contacts.copy()
    added = []
    for parent in big.tolist():
        joint_size = JOINT_SIZE * size[parent]
        paid_energy = float(joint_energy(joint_size))
        second, joint = len(kind), len(kind) + 1
        left_energy, left_size = energy[parent] - paid_energy, size[parent] - joint_size
        energy[parent], size[parent] = left_energy / 2, left_size / 2
        energy += [left_energy - left_energy / 2, paid_energy]
        size += [left_size - left_size / 2, joint_size]
        kind += [KIND_A, KIND_B]
        rate += [0.0, 0.0]
        _share_joints(contacts, parent, second, state.energy)
        added += [(parent, joint), (second, joint)]
    rows = np.vstack([contacts, np.array(added, dtype=np.int64)])
    return replace(state, kind=np.array(kind, dtype=np.uint8), energy=np.array(energy), size=np.array(size),
                   rate=np.array(rate), contacts=np.unique(rows, axis=0),
                   splits=state.splits + len(big))


def _share_joints(contacts: np.ndarray, parent: int, second: int, energy: np.ndarray) -> None:
    """Hand every other one of the parent's joints, ordered by energy then index, to the second half."""
    rows = np.flatnonzero(contacts[:, 0] == parent)
    joints = contacts[rows, 1]
    order = np.lexsort((joints, energy[joints]))
    contacts[rows[order[1::2]], 0] = second
