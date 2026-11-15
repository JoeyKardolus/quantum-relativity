"""New state: an irregular ABAB structure with slightly uneven energy, nothing placed by hand.

Contract: `new_state(volumes, seed, unevenness)` returns (State, positions):
  scatter   `volumes` points uniform in an open cube of volume `volumes` (seeded)
  kinds     a random half (volumes // 2) are energy volumes, the rest joints
  contacts  every point joins its NEIGHBOURS (6) nearest points of the other kind,
            and the union of those pairs is the contacts, so every A touches only
            B's and every B only A's, about 6 to 8 each; points on the outside are
            chosen by fewer and have fewer contacts (open, not wrapped)
  energy    per volume eps = 1 + unevenness * u, u uniform in [-1, 1] (seeded):
            an energy volume holds eps ** 4 at size 1 (the cells engine's start: an
            uneven energy density, an even size), a joint JOINT_ENERGY * eps and
            wants joint_rest_size(energy) (joint_rest_size.py)
  joints    each starts at its balance (joint_balance.py) under the start's squeeze
            (pull_potential.py, joint_squeeze.py, Settings.all_on()), rate 0
collapses 0, splits 0, start_size the sum of all sizes, start_mean_size the mean
energy volume's size (1). positions (volumes, 3) are for drawing only; the State
holds none and nothing reads them. Raises ValueError for unevenness outside [0, 1)
(energy must stay positive) or fewer than 2 * (NEIGHBOURS + 1) volumes.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from physics.abab.joint_balance import joint_balance
from physics.abab.joint_rest_size import JOINT_ENERGY, joint_rest_size
from physics.abab.joint_squeeze import joint_squeeze
from physics.abab.pull_potential import pull_potential
from physics.abab.settings import Settings
from physics.abab.state import KIND_A, KIND_B, State

NEIGHBOURS = 6


def new_state(volumes: int, seed: int, unevenness: float) -> tuple[State, np.ndarray]:
    """An irregular ABAB structure of `volumes` volumes, joints balanced, and its drawing positions."""
    if not 0 <= unevenness < 1:
        raise ValueError("unevenness must be in [0, 1) so every energy stays positive")
    if volumes < 2 * (NEIGHBOURS + 1):
        raise ValueError(f"need at least {2 * (NEIGHBOURS + 1)} volumes")
    generator = np.random.default_rng(seed)
    positions = generator.uniform(0.0, np.cbrt(volumes), size=(volumes, 3))
    kind = np.full(volumes, KIND_B, dtype=np.uint8)
    kind[generator.permutation(volumes)[:volumes // 2]] = KIND_A
    eps = 1 + unevenness * generator.uniform(-1.0, 1.0, size=volumes)
    is_volume = kind == KIND_A
    energy = np.where(is_volume, eps ** 4, JOINT_ENERGY * eps)
    rest = joint_rest_size(energy)
    size = np.where(is_volume, 1.0, rest)
    state = State(kind=kind, energy=energy, size=size, rate=np.zeros(volumes),
                  contacts=_contacts(positions, is_volume), collapses=0, splits=0,
                  start_size=0.0, start_mean_size=1.0)
    settings = Settings.all_on()
    squeeze = joint_squeeze(state, pull_potential(state), settings.attraction)
    size = np.where(is_volume, 1.0, joint_balance(rest, squeeze, settings))
    return replace(state, size=size, start_size=float(size.sum())), positions


def _contacts(positions: np.ndarray, is_volume: np.ndarray) -> np.ndarray:
    """Unique sorted (energy volume, joint) rows: each point to its NEIGHBOURS nearest of the other kind."""
    volume_ids, joint_ids = np.flatnonzero(is_volume), np.flatnonzero(~is_volume)
    _, nearest_joints = cKDTree(positions[joint_ids]).query(positions[volume_ids], k=NEIGHBOURS)
    _, nearest_volumes = cKDTree(positions[volume_ids]).query(positions[joint_ids], k=NEIGHBOURS)
    from_volumes = np.column_stack([np.repeat(volume_ids, NEIGHBOURS), joint_ids[nearest_joints.ravel()]])
    from_joints = np.column_stack([volume_ids[nearest_volumes.ravel()], np.repeat(joint_ids, NEIGHBOURS)])
    return np.unique(np.vstack([from_volumes, from_joints]), axis=0)
