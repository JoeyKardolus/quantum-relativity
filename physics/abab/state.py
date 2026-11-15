"""State: the ABAB structure as the physics sees it, kind, energy and size per volume and who touches whom.

Contract: frozen record; step() returns a new one and never changes an old one.
  kind               (n,) uint8 per volume: KIND_A (0) an energy volume, KIND_B (1) a joint
  energy             (n,) float per volume, > 0
  size               (n,) float per volume, > 0. An energy volume's size follows its
                     energy (grow_by_energy.py); a joint's size swings (rate below)
  rate               (n,) float per volume: a joint's size change per step (its
                     momentum, the swing); 0 for every energy volume
  contacts           (m, 2) int64 rows (a, b), a an energy volume and b a joint,
                     unique, sorted; every A touches only B's and every B only A's
  collapses, splits  joint collapses and volume splits since the start
  start_size         the sum of every volume's size at the start
  start_mean_size    the mean energy volume's size at the start (the split trigger's unit)
No positions: new_state() hands them back beside the State, for drawing only.
total_energy() is the sum of energy, which no rule changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KIND_A = 0
KIND_B = 1


@dataclass(frozen=True)
class State:
    """Kind, energy, size and swing rate per volume, the A-B contacts, event counts and the start's sizes."""

    kind: np.ndarray
    energy: np.ndarray
    size: np.ndarray
    rate: np.ndarray
    contacts: np.ndarray
    collapses: int
    splits: int
    start_size: float
    start_mean_size: float

    def total_energy(self) -> float:
        """The sum of every volume's energy."""
        return float(self.energy.sum())
