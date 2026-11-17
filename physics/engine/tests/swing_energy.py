"""Swing energy (test helper): the energy the joint swing must keep, written out on the host from the swing's own law.

Contract: `swing_energy(state, settings, load, swinging)` returns for universe 0
  E = 1/2 v^T M^+ v + sum_i U_i(s_i),  U_i(s) = push * wanted_i^p / ((p - 1) s^(p - 1)) - load_i * s
over the joints `swinging` (bool per cell id), p = settings.push_steepness, M the
clock-weighted graph Laplacian over those joints (two of them share a spring when they
share an energy volume, weight rate_i * rate_j / joint_inertia), M^+ its pseudo-inverse,
v and s their speeds and sizes (state.speed, state.size) and `load` the per-cell load
joint_swing was given (float64 (universes, capacity)). A joint outside `swinging` is in
no spring and adds nothing. Reads state.rate and state.wanted (cell_laws.py); changes
nothing. Dense, so for small universes only.
"""
from __future__ import annotations

import numpy as np

from physics.engine import Settings, State


def swing_energy(state: State, settings: Settings, load: np.ndarray, swinging: np.ndarray) -> float:
    """Kinetic plus push and load energy of the swinging joints of universe 0."""
    joints = np.flatnonzero(swinging)
    index = {int(joint): place for place, joint in enumerate(joints)}
    laplacian = np.zeros((joints.size, joints.size))
    for volume in np.flatnonzero(state.kind[0] == 1):
        members = [index[int(joint)] for joint in state.joints_of(0, int(volume)) if int(joint) in index]
        for first in members:
            for second in members:
                if first != second:
                    weight = state.rate[0, joints[first]] * state.rate[0, joints[second]] / settings.joint_inertia
                    laplacian[first, second] -= weight
                    laplacian[first, first] += weight
    speed = state.speed[0, joints]
    size = state.size[0, joints]
    steepness = settings.push_steepness
    moving = 0.5 * speed @ np.linalg.pinv(laplacian, hermitian=True) @ speed
    held = (settings.push * state.wanted[0, joints] ** steepness / ((steepness - 1.0) * size ** (steepness - 1.0))
            - load[0, joints] * size)
    return float(moving + held.sum())
