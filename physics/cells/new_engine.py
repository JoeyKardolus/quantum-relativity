"""New engine: an open cube of tetrahedral energy volumes with slightly uneven energy, its mesh cut once and settled.

Contract: `new_engine(cells, seed, unevenness)` returns an Engine (engine.py) of about
`cells` energy volumes (first_mesh.py: the tetrahedra whose centroid falls in a
cube of volume `cells`, after its outer slivers are dropped):
  energy     per energy volume, its measured start volume V times eps ** 4, with
             eps = 1 + unevenness * u its energy per quantum, u uniform in [-1, 1]
             (seeded); quanta V * eps ** 3, so rest_volume(energy, quanta) is V
             and the energy density eps ** 4 is uneven everywhere (as the earlier
             start's (1 +- unevenness) ** 4)
  volume     the measured volumes after settling; each target is the rest volume
             plus half its joint walls
  mesh       first_mesh(cells, rng), seeded: the only tessellation the engine runs;
             outer faces border nothing
  joints     one per triangle between two tetrahedra, at its balance thickness
             (joint_balance.py, EngineSettings.all_on()), rate 0; pull from the
             start's pull potential; seams the edges where two or more joint walls
             meet (mesh_seams.py)
  settle     SETTLE_PASSES (200) rounds of relax_corners() toward the targets, so the
             walls find room before step 1; only corners move; then the joints are
             balanced again on the measured volumes
collapses 0, splits 0, neighbour_changes 0, start_space the settled mesh volume.
Nothing is placed by hand. Raises ValueError when unevenness is outside [0, 1)
(energy must stay positive) or cells < 30.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.cells.cell_targets import cell_targets
from physics.cells.cell_volumes import cell_volumes
from physics.cells.engine import Engine
from physics.cells.engine_settings import EngineSettings
from physics.cells.face_joints import face_joints
from physics.cells.first_mesh import first_mesh
from physics.cells.grid import Grid
from physics.cells.joint_balance import joint_balance
from physics.cells.joint_squeeze import joint_squeeze
from physics.cells.mesh_seams import mesh_seams
from physics.cells.pull_potential import pull_potential
from physics.cells.relax_corners import relax_corners
from physics.cells.rest_volume import rest_volume
from physics.cells.seam_openness import seam_openness
from physics.cells.tet_faces import tet_faces

SETTLE_PASSES = 200


def new_engine(cells: int, seed: int, unevenness: float) -> Engine:
    """An engine of about `cells` tetrahedral energy volumes with slightly uneven energy and a settled mesh."""
    if not 0 <= unevenness < 1:
        raise ValueError("unevenness must be in [0, 1) so every energy stays positive")
    settings = EngineSettings.all_on()
    rng = np.random.default_rng(seed)
    corners, tets = first_mesh(cells, rng)
    count = len(tets)
    measured = cell_volumes(corners, tets)
    per_quantum = 1.0 + unevenness * rng.uniform(-1.0, 1.0, count)
    energy, quanta = measured * per_quantum ** 4, measured * per_quantum ** 3
    faces, face_cells = tet_faces(tets)
    joints, face_joint = face_joints(face_cells)
    seams = mesh_seams(faces, face_joint)
    grid = Grid(cell_count=count, joints=joints, energy=energy, quanta=quanta, volume=measured,
                thickness=np.zeros(len(joints)), rate=np.zeros(len(joints)),
                squeeze=np.zeros(len(joints)), pull=np.zeros(count), charge=np.zeros(count),
                seams=seams, seam_open=np.ones(int(seams[-1, 0]) + 1 if len(seams) else 0))
    balanced = _balanced(grid, settings)
    rest = rest_volume(energy, quanta)
    for _ in range(SETTLE_PASSES // 10):
        targets = cell_targets(rest, balanced.thickness, faces, face_cells, face_joint, corners)
        corners = relax_corners(corners, tets, targets, 10)
    settled = _balanced(replace(balanced, volume=cell_volumes(corners, tets)), settings)
    return Engine(grid=settled, corners=corners, tets=tets, faces=faces, face_cells=face_cells,
                  face_joint=face_joint, collapses=0, splits=0,
                  start_space=float(cell_volumes(corners, tets).sum()), neighbour_changes=0)


def _balanced(grid: Grid, settings: EngineSettings) -> Grid:
    """The grid with its joints at balance, its pull, squeeze and seam openness set."""
    balanced = replace(grid, thickness=joint_balance(grid, settings),
                       pull=settings.attraction * pull_potential(grid) / 2)
    balanced = replace(balanced, squeeze=joint_squeeze(balanced, settings))
    return replace(balanced, seam_open=seam_openness(balanced, settings))
