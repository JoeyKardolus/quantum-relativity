"""Grid: the woven grid of energy volumes and the joints between them; no shapes, no coordinates.

Contract: a frozen record of
  cell_count   number of energy volumes n
  joints       (k, 2) int, each row the two neighbouring energy volumes a joint separates
  energy       (n,) energy held inside each energy volume
  volume       (n,) size of each energy volume
  squeeze      (k,) how hard each joint was squeezed in the last step (0 at the start)
  merged_into  None at the start; after a step, merged_into[old_cell] = that cell's index now
total_energy() is the sum of energy, the quantity joint collapse and energy flow keep.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Energy volumes (energy, volume) and the joints between neighbours (pairs, squeeze)."""

    cell_count: int
    joints: np.ndarray
    energy: np.ndarray
    volume: np.ndarray
    squeeze: np.ndarray
    merged_into: np.ndarray | None = None

    def total_energy(self) -> float:
        """Sum of the energy held by every energy volume."""
        return float(self.energy.sum())
