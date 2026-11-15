"""Grid: the woven grid of energy volumes and the joints between them; no shapes, no coordinates.

Contract: a frozen record of
  cell_count    number of energy volumes n (the mesh's tetrahedra, engine.py)
  joints        (k, 2) int, a < b, sorted: the two neighbouring energy volumes each joint
                separates (the pairs of the mesh's shared triangles); every energy
                volume has at most 4, one per face, fewer on the outside
  energy        (n,) energy held inside each energy volume
  quanta        (n,) quanta of space each energy volume holds (rest_volume.py): the
                energy per quantum sets its size; a split shares them by volume like
                energy, a collapse hands them on with the energy, the leak never
                moves them
  volume        (n,) each energy volume's measured size: its tetrahedron's volume in
                the mesh (cell_volumes.py), which follows rest_volume(energy, quanta)
                (rule 4) as the corners relax, as far as the shared corners allow
  thickness     (k,) each joint's thickness, a length: the wall between its two volumes
  rate          (k,) each joint's rate of change of thickness per step (its momentum:
                joints swing, swing_joints.py)
  squeeze       (k,) derived, for drawing: 0 at or above rest thickness, 1 at the
                collapse limit (joint_squeeze.py): how hard the joint pushes
  pull          (n,) derived, for drawing: the pull each energy volume exerts,
                attraction * pull_potential / 2 (step.py)
  charge        (n,) per energy volume, zeros for now (what charge is, is still open)
  seams         (r, 2) int rows (seam, joint): every joint around each seam, a mesh
                edge where two or more joint walls meet (mesh_seams.py); the leak
                flows only along seams (leak.py). The volumes around a seam are the
                cells of its joints.
  seam_open     (m,) derived: each seam's openness, 1 relaxed, 0 sealed (seam_openness.py)
total_energy() is the sum of energy, which no rule changes (the leak moves it,
collapse hands it on, split shares it out). seam_count() is m.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Energy volumes (energy, quanta, volume, charge), the joints between them (thickness, rate), seams."""

    cell_count: int
    joints: np.ndarray
    energy: np.ndarray
    quanta: np.ndarray
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

    def seam_count(self) -> int:
        """Number of seams: one more than the highest seam number in seams."""
        return int(self.seams[-1, 0]) + 1 if len(self.seams) else 0
