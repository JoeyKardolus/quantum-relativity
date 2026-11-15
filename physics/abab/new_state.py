"""New state: a 3D checkerboard grid of energy volumes and joints with slightly uneven energy.

Contract: `new_state(volumes, seed, unevenness)` returns (State, positions):
  grid      side = round(cbrt(volumes)) (at least 3), side**3 volumes on a cubic grid
            (owner, 2026-11-15: start from an actual grid; the unevenness comes from the
            energy, not from placement)
  kinds     the 3D checkerboard: (i + j + k) even is an energy volume, odd a joint
  contacts  every volume joins its face neighbours on the grid (up to 6; fewer on the
            outside, open, not wrapped), which are always of the other kind, so every
            A touches only B's and every B only A's
  energy    per volume eps = 1 + unevenness * u, u uniform in [-1, 1] (seeded):
            an energy volume holds eps ** 4 at size 1 (the cells engine's start: an
            uneven energy density, an even size), a joint JOINT_ENERGY * eps and
            wants joint_rest_size(energy) (joint_rest_size.py)
  joints    each starts at its balance (joint_balance.py) under the start's squeeze
            (pull_potential.py, joint_squeeze.py, Settings.all_on()), rate 0
collapses 0, splits 0, start_size the sum of all sizes, start_mean_size the mean
energy volume's size (1). positions (volumes, 3) are for drawing only; the State
holds none and nothing reads them. Raises ValueError for unevenness outside [0, 1)
(energy must stay positive).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.abab.joint_balance import joint_balance
from physics.abab.joint_rest_size import JOINT_ENERGY, joint_rest_size
from physics.abab.joint_squeeze import joint_squeeze
from physics.abab.pull_potential import pull_potential
from physics.abab.settings import Settings
from physics.abab.state import KIND_A, KIND_B, State


def new_state(volumes: int, seed: int, unevenness: float) -> tuple[State, np.ndarray]:
    """An irregular ABAB structure of `volumes` volumes, joints balanced, and its drawing positions."""
    if not 0 <= unevenness < 1:
        raise ValueError("unevenness must be in [0, 1) so every energy stays positive")
    generator = np.random.default_rng(seed)
    side = max(3, int(round(np.cbrt(volumes))))
    index = np.indices((side, side, side)).reshape(3, -1).T
    positions = index.astype(float)
    volumes = len(index)
    kind = np.where(index.sum(axis=1) % 2 == 0, KIND_A, KIND_B).astype(np.uint8)
    eps = 1 + unevenness * generator.uniform(-1.0, 1.0, size=volumes)
    is_volume = kind == KIND_A
    energy = np.where(is_volume, eps ** 4, JOINT_ENERGY * eps)
    rest = joint_rest_size(energy)
    size = np.where(is_volume, 1.0, rest)
    state = State(kind=kind, energy=energy, size=size, rate=np.zeros(volumes),
                  contacts=_grid_contacts(index, side, is_volume), collapses=0, splits=0,
                  start_size=0.0, start_mean_size=1.0)
    settings = Settings.all_on()
    squeeze = joint_squeeze(state, pull_potential(state), settings.attraction)
    size = np.where(is_volume, 1.0, joint_balance(rest, squeeze, settings))
    return replace(state, size=size, start_size=float(size.sum())), positions


def _grid_contacts(index: np.ndarray, side: int, is_volume: np.ndarray) -> np.ndarray:
    """Unique sorted (energy volume, joint) rows: every pair of face neighbours on the grid."""
    flat = (index[:, 0] * side + index[:, 1]) * side + index[:, 2]
    rows = []
    for axis, stride in enumerate((side * side, side, 1)):
        here = flat[index[:, axis] < side - 1]
        rows.append(np.column_stack([here, here + stride]))
    pairs = np.vstack(rows)
    first_is_volume = is_volume[pairs[:, 0]]
    ordered = np.where(first_is_volume[:, None], pairs, pairs[:, ::-1])
    return np.unique(ordered, axis=0)
