"""Grid: the woven grid of energy volumes and the joints between them; no shapes, no coordinates.

Contract: a frozen record of
  cell_count    number of energy volumes n
  joints        (k, 2) int, each row the two neighbouring energy volumes a joint separates
  energy        (n,) energy held inside each energy volume
  volume        (n,) size of each energy volume
  joint_volume  (k,) size of each joint: the pushing volume between its two energy volumes
  squeeze       (k,) derived, for drawing: 0 for a joint at or above its rest volume,
                1 for a joint at its collapse limit (step.py gives the formula)
  charge        (n,) per energy volume, zeros for now (what charge is, is still open)
  seams         (m, 3) int, a < b < c: the three energy volumes around each seam, the
                edge where their three joint walls meet; the leak flows only along
                seams (leak.py). The three pairs of every seam are joints.
  seam_open     (m,) derived, for drawing and engine_stats: each seam's openness,
                1 relaxed, 0 sealed (seam_openness.py)
  merged_into   None at the start; after a step, merged_into[old_cell] = that cell's index now
total_energy() is the sum of energy, which no rule changes (the leak moves it, collapse adds it).
total_space() is the sum of energy volumes plus the sum of joint volumes: all of space,
since energy volumes and joints fill it with no gaps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Energy volumes (energy, volume, charge), the joints between neighbours, and their seams."""

    cell_count: int
    joints: np.ndarray
    energy: np.ndarray
    volume: np.ndarray
    joint_volume: np.ndarray
    squeeze: np.ndarray
    charge: np.ndarray
    seams: np.ndarray
    seam_open: np.ndarray
    merged_into: np.ndarray | None = None

    def total_energy(self) -> float:
        """Sum of the energy held by every energy volume."""
        return float(self.energy.sum())

    def total_space(self) -> float:
        """Sum of every energy volume's and every joint's volume."""
        return float(self.volume.sum() + self.joint_volume.sum())
