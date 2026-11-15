"""Stats: what the ABAB structure looks like now.

Contract: `stats(state, settings)` returns a dict:
  energy_volumes  energy volumes (A) now
  joints          joints (B) now
  contacts        A-B contacts now
  collapses       joint collapses since the start
  splits          volume splits since the start
  too_big         energy volumes above the split trigger now (split_size times
                  start_mean_size) that are too dilute to pay for a joint (split_volumes.py)
  size_vs_start   the sum of every volume's size now / at the start
  total_energy    sum of all energy (never changes)
  lumps           lumps now (lumps.py)
  sealed_lumps    lumps whose outer joints are all closed (lumps.py)
settings (default Settings.all_on()) gives the seal point the openness is read with.
"""
from __future__ import annotations

import numpy as np

from physics.abab.joint_openness import joint_openness
from physics.abab.joint_rest_size import joint_rest_size
from physics.abab.lumps import lumps
from physics.abab.settings import Settings
from physics.abab.state import KIND_A, KIND_B, State


def stats(state: State, settings: Settings | None = None) -> dict:
    """Energy volumes, joints, contacts, collapses, splits, too big, size against the start, energy, lumps, sealed lumps."""
    settings = settings or Settings.all_on()
    is_joint = state.kind == KIND_B
    openness = np.where(is_joint, joint_openness(state.size, joint_rest_size(state.energy), settings), 0.0)
    _, sealed = lumps(state, openness)
    return {"energy_volumes": int((state.kind == KIND_A).sum()),
            "joints": int(is_joint.sum()),
            "contacts": len(state.contacts),
            "collapses": state.collapses,
            "splits": state.splits,
            "too_big": int(((state.kind == KIND_A)
                            & (state.size > settings.split_size * state.start_mean_size)).sum()),
            "size_vs_start": float(state.size.sum()) / state.start_size,
            "total_energy": state.total_energy(),
            "lumps": len(sealed),
            "sealed_lumps": int(sealed.sum())}
