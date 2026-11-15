"""Cells layer: the engine. Energy volumes and joints, the rules that step them, and the
mesh of shared corners that is their exact, gap-free geometry in an open cube.

The rules and every measurement read only energies, sizes, joint thicknesses and the
neighbour graph (Grid). The mesh (corners shared between cells, faces shared between
the two cells of a joint) is cut once from a Voronoi tessellation at the start; after
that a step only moves corners, and the topology changes only by joint collapse (two
cells become one) and volume split (a cell stretched too thin is cut in two).

Module map:
  engine.py             Engine: Grid + corners, faces, face_cells, face_joint, fan, counts
  engine_settings.py    EngineSettings: rule strengths, all_on() defaults
  new_engine.py         new_engine: seeded uneven energy, first mesh, joints balanced, settled
  step.py               step: pure Engine -> Engine; pull, swing, leak, collapse, split, corners
  engine_stats.py       engine_stats: cells, joints, collapses, splits, size, energy, lumps
  mesh_message.py       mesh_message: the topology for the page, on start, collapse, split
  frame_arrays.py       frame_arrays: corners float32, per-cell pull and per-joint push uint8
  lumps.py              lumps: lump number per energy volume, and which lumps have sealed
  grid.py               Grid: per energy volume energy, volume, pull, charge; per joint
                        thickness, rate, squeeze; seams; no shapes, no coordinates
  rest_volume.py        rest_volume: the energy-volume rule, volume = (VOLUME_SCALE / E) ** 3
  joint_rest.py         joint_rest: a joint's rest thickness, a share of its volumes' length
  joint_push.py         joint_push: a joint's push toward rest, steep as it thins
  joint_balance.py      joint_balance: the thickness where push cancels squeeze
  joint_squeeze.py      joint_squeeze: 0..1 how close each joint is to its limit (its push)
  swing_joints.py       swing_joints: one undamped spring step of every joint
  pull_potential.py     pull_potential: grounded graph-Poisson spread of energy
  seam_joints.py        seam_joints: the three joints that meet at each seam
  seam_openness.py      seam_openness: 1 relaxed .. 0 at the seal point, per seam
  leak.py               leak: each volume leaks energy along open seams, by pull
  first_mesh.py         first_mesh: the one Voronoi tessellation, shared corners (pyvoro2)
  fan.py                Fan: faces as triangles with (cell, corner) gradient slots
  build_fan.py          build_fan: the Fan of a topology
  cell_volumes.py       cell_volumes: each cell's measured volume
  face_areas.py         face_areas: each face's area
  cell_targets.py       cell_targets: energy volume plus half its joint walls, per cell
  relax_corners.py      relax_corners: corners move so cells follow their targets
  face_joints.py        face_joints: joints from the faces, thickness carried over
  mesh_seams.py         mesh_seams: three-cell edges of the mesh
  collapse_cells.py     collapse_cells: the mesh after joints give way
  split_cell.py         split_cell: one cell cut in two by a new face
  drop_loose_corners.py drop_loose_corners: corners no face uses go
"""
from physics.cells.grid import Grid
from physics.cells.rest_volume import VOLUME_SCALE, rest_volume
from physics.cells.joint_rest import joint_rest
from physics.cells.engine_settings import EngineSettings
from physics.cells.pull_potential import pull_potential
from physics.cells.joint_push import joint_push
from physics.cells.joint_balance import joint_balance
from physics.cells.joint_squeeze import joint_squeeze
from physics.cells.swing_joints import swing_joints
from physics.cells.seam_joints import seam_joints
from physics.cells.seam_openness import seam_openness
from physics.cells.leak import leak
from physics.cells.fan import Fan
from physics.cells.build_fan import build_fan
from physics.cells.cell_volumes import cell_volumes
from physics.cells.face_areas import face_areas
from physics.cells.cell_targets import cell_targets
from physics.cells.relax_corners import relax_corners
from physics.cells.first_mesh import first_mesh
from physics.cells.face_joints import face_joints
from physics.cells.mesh_seams import mesh_seams
from physics.cells.collapse_cells import collapse_cells
from physics.cells.split_cell import split_cell
from physics.cells.drop_loose_corners import drop_loose_corners
from physics.cells.engine import Engine
from physics.cells.new_engine import new_engine
from physics.cells.step import step
from physics.cells.lumps import lumps
from physics.cells.engine_stats import engine_stats
from physics.cells.mesh_message import mesh_message
from physics.cells.frame_arrays import frame_arrays

__all__ = ["Grid", "VOLUME_SCALE", "rest_volume", "joint_rest", "EngineSettings", "pull_potential",
           "joint_push", "joint_balance", "joint_squeeze", "swing_joints", "seam_joints", "seam_openness", "leak",
           "Fan", "build_fan", "cell_volumes", "face_areas", "cell_targets", "relax_corners",
           "first_mesh", "face_joints", "mesh_seams", "collapse_cells", "split_cell",
           "drop_loose_corners", "Engine", "new_engine", "step", "lumps", "engine_stats",
           "mesh_message", "frame_arrays"]
