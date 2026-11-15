"""Grid: the woven grid of energy volumes and the joints between them; no shapes, no coordinates.

Contract: a frozen record of
  cell_count    number of energy volumes n
  joints        (k, 2) int, a < b, sorted: the two neighbouring energy volumes each joint
                separates (the pairs of the mesh's shared faces, engine.py)
  energy        (n,) energy held inside each energy volume
  volume        (n,) each energy volume's measured size: its cell's volume in the mesh
                (cell_volumes.py), which follows rest_volume(energy) (rule 4) as the
                corners relax; a cell just split or collapsed reaches it over steps
  thickness     (k,) each joint's thickness, a length: the wall between its two volumes
  rate          (k,) each joint's rate of change of thickness per step (its momentum:
                joints swing, swing_joints.py)
  squeeze       (k,) derived, for drawing: 0 at or above rest thickness, 1 at the
                collapse limit (joint_squeeze.py): how hard the joint pushes
  pull          (n,) derived, for drawing: the pull each energy volume exerts,
                attraction * pull_potential / 2 (step.py)
  charge        (n,) per energy volume, zeros for now (what charge is, is still open)
  seams         (m, 3) int, a < b < c: the three energy volumes around each seam, a mesh
                edge where their three joint walls meet; the leak flows only along
                seams (leak.py). The three pairs of every seam are joints.
  seam_open     (m,) derived: each seam's openness, 1 relaxed, 0 sealed (seam_openness.py)
total_energy() is the sum of energy, which no rule changes (the leak moves it,
collapse adds it together, split shares it out).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Energy volumes (energy, volume, charge), the joints between them (thickness, rate), seams."""

    cell_count: int
    joints: np.ndarray
    energy: np.ndarray
    volume: np.ndarray
    thickness: np.ndarray
    rate: np.ndarray
    squeeze: np.ndarray
    pull: np.ndarray
    charge: np.ndarray
    seams: np.ndarray
    seam_open: np.ndarray

    def total_energy(self) -> float:
        """Sum of the energy held by every energy volume."""
        return float(self.energy.sum())
