"""New engine: an open cube of energy volumes and joint cells with slightly uneven energy, cut once and settled.

Contract: `new_engine(cells, seed, unevenness)` returns an Engine (engine.py) of about
`cells` energy volumes (first_mesh.py: the skeleton tetrahedra whose centroid falls in
a cube of volume `cells`, after its outer slivers are dropped):
  cut        joint_cells.py: each skeleton tetrahedron gets an energy volume with its
             own corners, pulled toward its centre; between them joint cells (one per
             inner skeleton face), seam cells (one per skeleton edge) and corner cells
             (one per skeleton corner), filling space with no gaps (place_corners.py)
  energy     per energy volume, its measured size V times eps ** 4, with
             eps = 1 + unevenness * u its energy per quantum, u uniform in [-1, 1]
             (seeded); quanta V * eps ** 3, so rest_volume(energy, quanta) is V
             and the energy density eps ** 4 is uneven everywhere
  joints     one per inner skeleton face, at its balance thickness (joint_balance.py,
             EngineSettings.all_on()), rate 0; the corners are placed for those
             thicknesses, the sizes measured and the balance found again,
             BALANCE_ROUNDS (3) times (the thickness is a share of the sizes); pull
             from the start's pull potential; seams the seam cells two or more joint
             cells touch (mesh_seams.py)
  settle     SETTLE_PASSES (20) rounds of relax_corners() toward the targets
             (cell_targets.py), so the joint cells take their thickness times their
             area before step 1; only corners move; then the joints are balanced
             again on the measured sizes
collapses 0, splits 0, neighbour_changes 0, start_space the settled volume of all
cells, start_mean_volume the mean energy volume.
Nothing is placed by hand. Raises ValueError when unevenness is outside [0, 1)
(energy must stay positive) or cells < 30.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.cells.cell_sizes import cell_sizes
from physics.cells.cell_targets import cell_targets
from physics.cells.cell_volumes import cell_volumes
from physics.cells.engine import Engine
from physics.cells.engine_settings import EngineSettings
from physics.cells.first_mesh import first_mesh
from physics.cells.grid import Grid
from physics.cells.joint_balance import joint_balance
from physics.cells.joint_cells import joint_cells
from physics.cells.joint_offsets import joint_offsets
from physics.cells.joint_rest import joint_rest
from physics.cells.joint_squeeze import joint_squeeze
from physics.cells.mesh_seams import mesh_seams
from physics.cells.place_corners import place_corners
from physics.cells.pull_potential import pull_potential
from physics.cells.relax_corners import relax_corners
from physics.cells.seam_openness import seam_openness

BALANCE_ROUNDS = 3
SETTLE_PASSES = 20


def new_engine(cells: int, seed: int, unevenness: float) -> Engine:
    """An engine of about `cells` energy volumes with joint cells between them, slightly uneven energy, settled."""
    if not 0 <= unevenness < 1:
        raise ValueError("unevenness must be in [0, 1) so every energy stays positive")
    settings = EngineSettings.all_on()
    rng = np.random.default_rng(seed)
    skeleton, tets = first_mesh(cells, rng)
    count = len(tets)
    per_quantum = 1.0 + unevenness * rng.uniform(-1.0, 1.0, count)
    woven = joint_cells(tets)
    cell_count = len(woven.cell_kind)
    seams = mesh_seams(woven.face_cells, woven.face_joint, woven.cell_kind)
    thickness = joint_rest(cell_volumes(skeleton, tets), woven.joints, settings.joint_rest)
    for _ in range(BALANCE_ROUNDS):
        corners = _placed(woven, skeleton, thickness, settings)
        measured = cell_sizes(corners, woven.faces, woven.face_cells, cell_count)[:count]
        grid = Grid(cell_count=count, joints=woven.joints, energy=measured * per_quantum ** 4,
                    quanta=measured * per_quantum ** 3, volume=measured, thickness=thickness,
                    rate=np.zeros(len(woven.joints)), squeeze=np.zeros(len(woven.joints)),
                    pull=np.zeros(count), charge=np.zeros(count), seams=seams,
                    seam_open=np.ones(int(seams[-1, 0]) + 1 if len(seams) else 0))
        thickness = joint_balance(grid, settings)
    grid = replace(grid, thickness=thickness)
    balanced = _balanced(grid, settings)
    corners = _placed(woven, skeleton, balanced.thickness, settings)
    built = cell_sizes(corners, woven.faces, woven.face_cells, cell_count)
    for _ in range(SETTLE_PASSES // 10):
        targets = cell_targets(balanced.volume, balanced.thickness, corners, woven.faces, woven.face_cells,
                               woven.cell_kind, built)
        corners = relax_corners(corners, woven.faces, woven.face_cells, targets, 10)
    sizes = cell_sizes(corners, woven.faces, woven.face_cells, cell_count)
    settled = _balanced(replace(balanced, volume=sizes[:count]), settings)
    return Engine(grid=settled, corners=corners, woven=woven, skeleton_corners=skeleton, collapses=0,
                  splits=0, start_space=float(sizes.sum()), start_mean_volume=float(sizes[:count].mean()),
                  neighbour_changes=0)


def _placed(woven, skeleton: np.ndarray, thickness: np.ndarray, settings: EngineSettings) -> np.ndarray:
    """Every corner where place_corners.py puts it for these joint thicknesses."""
    offsets = joint_offsets(skeleton, woven.skeleton_tets, woven.slot_joint, thickness, settings.joint_rest)
    return place_corners(woven.keys, skeleton, woven.skeleton_tets, offsets)


def _balanced(grid: Grid, settings: EngineSettings) -> Grid:
    """The grid with its joints at balance, its pull, squeeze and seam openness set."""
    balanced = replace(grid, thickness=joint_balance(grid, settings),
                       pull=settings.attraction * pull_potential(grid) / 2)
    balanced = replace(balanced, squeeze=joint_squeeze(balanced, settings))
    return replace(balanced, seam_open=seam_openness(balanced, settings))
