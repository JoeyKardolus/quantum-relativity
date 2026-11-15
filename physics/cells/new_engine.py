"""New engine: an open cube of energy volumes with slightly uneven energy, its mesh cut once and settled.

Contract: `new_engine(cells, seed, unevenness)` returns an Engine (engine.py) of
exactly `cells` energy volumes:
  energy     1 + unevenness * u, u uniform in [-1, 1] per energy volume (seeded)
  volume     the cells' measured volumes after settling; each cell's target is
             rest_volume(energy) (the energy-volume rule) plus half its joint walls
  mesh       the power cells of points uniform at random (seeded) in a cube of side
             (sum rest volume) ** (1/3), not wrapped, fitted to the rest volumes
             (first_mesh.py): the only tessellations the engine runs; walls of the
             cube border nothing
  joints     one per pair of cells sharing a face, at its balance thickness
             (joint_balance.py, EngineSettings.all_on()), rate 0; pull from the start's
             pull potential; seams the mesh's
             three-cell edges (mesh_seams.py)
  settle     SETTLE_PASSES (200) rounds of relax_corners() toward the cells' targets,
             so each cell's size matches its energy before step 1; only corners move;
             then the joints are balanced again on the measured volumes
collapses 0, splits 0, neighbour_changes 0, start_space the settled mesh volume.
Nothing is placed by hand. Raises ValueError when unevenness is outside [0, 1)
(energy must stay positive) or cells < 30.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.cells.build_fan import build_fan
from physics.cells.cell_targets import cell_targets
from physics.cells.cell_volumes import cell_volumes
from physics.cells.engine import Engine
from physics.cells.engine_settings import EngineSettings
from physics.cells.face_areas import face_areas
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

SETTLE_PASSES = 200


def new_engine(cells: int, seed: int, unevenness: float) -> Engine:
    """An engine of `cells` energy volumes with slightly uneven energy and a settled mesh."""
    if not 0 <= unevenness < 1:
        raise ValueError("unevenness must be in [0, 1) so every energy stays positive")
    if cells < 30:
        raise ValueError("ask for at least 30 cells so the cube has an inside")
    settings = EngineSettings.all_on()
    rng = np.random.default_rng(seed)
    energy = 1.0 + unevenness * rng.uniform(-1.0, 1.0, cells)
    volume = rest_volume(energy)
    side = float(np.cbrt(volume.sum()))
    corners, faces, face_cells = first_mesh(rng.random((cells, 3)) * side, side, volume)
    fan = build_fan(faces, face_cells, cells)
    no_origin = np.full(len(faces), -1, dtype=np.int64)
    joints, face_joint, _, _, _ = face_joints(face_cells, no_origin, face_areas(corners, fan),
                                              np.zeros(0), np.zeros(0))
    seams = mesh_seams(faces, face_cells)
    grid = Grid(cell_count=cells, joints=joints, energy=energy, volume=volume,
                thickness=np.zeros(len(joints)), rate=np.zeros(len(joints)),
                squeeze=np.zeros(len(joints)), pull=np.zeros(cells), charge=np.zeros(cells),
                seams=seams, seam_open=np.ones(len(seams)))
    balanced = replace(grid, thickness=joint_balance(grid, settings),
                       pull=settings.attraction * pull_potential(grid) / 2)
    balanced = replace(balanced, squeeze=joint_squeeze(balanced, settings))
    balanced = replace(balanced, seam_open=seam_openness(balanced, settings))
    for _ in range(SETTLE_PASSES // 10):
        corners = relax_corners(corners, fan, cell_targets(volume, balanced.thickness, face_joint,
                                                           corners, fan), 10)
    settled = replace(balanced, volume=cell_volumes(corners, fan))
    settled = replace(settled, thickness=joint_balance(settled, settings),
                      pull=settings.attraction * pull_potential(settled) / 2)
    settled = replace(settled, squeeze=joint_squeeze(settled, settings))
    settled = replace(settled, seam_open=seam_openness(settled, settings))
    return Engine(grid=settled, corners=corners, faces=tuple(faces), face_cells=face_cells,
                  face_joint=face_joint, fan=fan, collapses=0, splits=0,
                  start_space=float(cell_volumes(corners, fan).sum()), neighbour_changes=0)
