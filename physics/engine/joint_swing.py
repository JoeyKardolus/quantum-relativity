"""Joint swing: every joint's size moves like a spring with inertia, pushed by its own push against its neighbours'.

Contract: `joint_swing(state, settings, pull)` advances every joint's size and speed
(state.size, state.speed) by one step_time of its own local clock (state.rate). A
joint's neighbours are the joints it shares an energy volume with (the cell graph,
state.between). Its load is its push (cell_laws.py) plus the mean pull over its boxes
(`pull`, cell_count.py; the pull is at most 0 where energy is, so it squeezes the joint:
bible rule 8, "pull squeezes it and its push swells it back"). Its speed changes by
sum over neighbours (own load - their load) / joint_inertia per unit of its local time,
and its size by its speed: the joints are the springs of the grid, they overshoot and
swing round their balance (bible rule 5), and what one joint gains its neighbours lose,
so the joints' summed size is kept. Leapfrog, cut into as many equal substeps as the
stiffest joint needs (numerics: omega * substep <= 1/2), the push recomputed each
substep. Energy volumes are not moved (their size follows their energy). Returns the
substep count used (the largest over the batch).
"""
from __future__ import annotations

import math

import numpy as np

from physics.engine.settings import Settings
from physics.engine.state import State


def joint_swing(state: State, settings: Settings, pull: np.ndarray) -> int:
    """Swing every joint's size one step on its local clock; returns the substeps taken."""
    pairs = _pairs(state)
    joints = state.kind == -1
    load_pull = np.where(joints, pull, 0.0)
    degree = np.zeros(state.kind.shape)
    np.add.at(degree, (pairs[:, 0], pairs[:, 1]), 1.0)
    stiffness = degree * settings.push_steepness * state.cell_push / np.maximum(state.size, 1e-6)
    omega = math.sqrt(float(np.max(stiffness * state.rate ** 2, initial=0.0)) / settings.joint_inertia)
    count = max(1, math.ceil(settings.step_time * omega / 0.5))
    span = settings.step_time / count
    clocked = state.rate * span
    for _ in range(count):
        load = np.where(joints, _push(state, settings) + load_pull, 0.0)
        force = np.zeros(state.kind.shape)
        np.add.at(force, (pairs[:, 0], pairs[:, 1]), load[pairs[:, 0], pairs[:, 1]] - load[pairs[:, 0], pairs[:, 2]])
        state.speed = np.where(joints, state.speed + clocked * force / settings.joint_inertia, 0.0)
        state.size = np.where(joints, np.maximum(state.size + clocked * state.speed, 1e-3), state.size)
    state.cell_push = np.where(joints, _push(state, settings), 0.0)
    return count


def _push(state: State, settings: Settings) -> np.ndarray:
    """push * (wanted / size) ** steepness per cell."""
    return settings.push * (state.wanted / np.maximum(state.size, 1e-6)) ** settings.push_steepness


def _pairs(state: State) -> np.ndarray:
    """(universe, joint, neighbour joint) rows, both ways, for every two joints sharing an energy volume."""
    rows = []
    for universe in range(state.universes):
        joints = np.flatnonzero(state.kind[universe] == -1)
        sides = state.between[universe, joints]
        member = np.concatenate([joints, joints])
        volume = np.concatenate([sides[:, 0], sides[:, 1]])
        keep = volume >= 0
        member, volume = member[keep], volume[keep]
        order = np.argsort(volume, kind="stable")
        member, volume = member[order], volume[order]
        starts = np.flatnonzero(np.r_[True, volume[1:] != volume[:-1]])
        ends = np.r_[starts[1:], volume.size]
        for first, last in zip(starts, ends):
            group = member[first:last]
            if group.size > 1:
                one, two = np.meshgrid(group, group, indexing="ij")
                off = one != two
                rows.append(np.stack([np.full(int(off.sum()), universe), one[off], two[off]], axis=1))
    return np.concatenate(rows) if rows else np.zeros((0, 3), dtype=np.int64)
