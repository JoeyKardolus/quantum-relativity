"""Step: one step of the ABAB structure; the rules run on kind, energy, size and contacts only.

Contract: `step(state, settings)` is pure and deterministic, returning a new State:
  1. pull      p = pull_potential(state); each joint is squeezed by attraction times
               its energy volumes' mean p (joint_squeeze.py, rule 7)
  2. open      each joint's openness from its size at the start of the step against
               the size its energy wants (joint_openness.py, joint_rest_size.py)
  3. swing     every joint's size swings one step under push and squeeze, undamped
               (physics.abab.swing_joints: velocity Verlet in substeps from each
               swing's thinnest turning point; the push a hard wall)
  4. leak      energy moves A -> B -> A by pull through the open joints (leak.py);
               each energy volume's size follows its new energy (grow_by_energy.py)
  5. collapse  joints below push_limit times their rest size vanish and their energy
               volumes fall into one (collapse_joints.py)
  6. split     energy volumes above split_size times start_mean_size split in two
               around a new joint (split_volumes.py)
Total energy is changed by no rule: the leak moves it, collapse hands it on, split
shares it. Positions are never read.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.abab.collapse_joints import collapse_joints
from physics.abab.grow_by_energy import grow_by_energy
from physics.abab.joint_openness import joint_openness
from physics.abab.joint_rest_size import joint_rest_size
from physics.abab.joint_squeeze import joint_squeeze
from physics.abab.leak import leak
from physics.abab.pull_potential import pull_potential
from physics.abab.settings import Settings
from physics.abab.split_volumes import split_volumes
from physics.abab.state import KIND_B, State
from physics.abab.swing_joints import swing_joints


def step(state: State, settings: Settings) -> State:
    """Pull, openness, swing, leak, then collapse and split."""
    is_joint = state.kind == KIND_B
    potential = pull_potential(state)
    squeeze = joint_squeeze(state, potential, settings.attraction)
    rest = np.where(is_joint, joint_rest_size(state.energy), state.size)
    openness = np.where(is_joint, joint_openness(state.size, rest, settings), 0.0)
    size, rate = state.size.copy(), state.rate.copy()
    size[is_joint], rate[is_joint] = swing_joints(state.size[is_joint], state.rate[is_joint], rest[is_joint],
                                                  squeeze[is_joint], settings.joint_push, settings.push_power)
    energy = leak(state, np.maximum(potential, 0.0), openness, settings.leak)
    size = np.where(is_joint, size, grow_by_energy(size, state.energy, energy))
    moved = replace(state, energy=energy, size=size, rate=rate)
    collapsed = collapse_joints(moved, is_joint & (size < settings.push_limit * rest))
    return split_volumes(collapsed, settings.split_size * state.start_mean_size)
