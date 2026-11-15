"""Cells layer: the engine. Energy volumes and joint cells, the rules that step them, and the
woven cells that are their exact, gap-free geometry in an open cube.

The rules and every measurement read only energies, quanta, sizes, joint thicknesses
and the neighbour graph (Grid). A skeleton of tetrahedra (as in causal dynamical
triangulations) is cut once by Delaunay at the start and says which cells neighbour
which; from it the cells are cut (joint_cells.py): every energy volume a tetrahedron
with its own four corners, every joint a cell of its own on the face between two
energy volumes, and seam and corner cells of joint material where joints meet. After
that a step only moves corners (every corner), and the topology changes only by joint
collapse (an edge of the joint's skeleton face drawn into one corner, its tetrahedra
falling in) and volume split (the 1-4 move), after which the cells are cut again.

Module map:
  engine.py             Engine: Grid + corners, woven cells, skeleton, counts
  engine_settings.py    EngineSettings: rule strengths, all_on() defaults
  new_engine.py         new_engine: seeded uneven energy, first mesh, joints balanced, settled
  step.py               step: pure Engine -> Engine; pull, swing, leak, collapse, split, corners
  engine_stats.py       engine_stats: cells, joints, collapses, splits, size, energy, lumps
  mesh_message.py       mesh_message: the topology for the page, on start, collapse, split
  frame_arrays.py       frame_arrays: corners float32, per-cell pull and per-joint push uint8
  lumps.py              lumps: lump number per energy volume, and which lumps have sealed
  grid.py               Grid: per energy volume energy, volume, pull, charge; per joint
                        thickness, rate, squeeze; seams; no shapes, no coordinates
  rest_volume.py        rest_volume: the energy-volume rule, size from energy per quantum
  joint_rest.py         joint_rest: a joint's rest thickness, a share of its volumes' length
  joint_push.py         joint_push: a joint's push toward rest, steep as it thins
  joint_balance.py      joint_balance: the thickness where push cancels squeeze
  joint_squeeze.py      joint_squeeze: 0..1 how close each joint is to its limit (its push)
  swing_joints.py       swing_joints: one undamped spring step of every joint
  pull_potential.py     pull_potential: grounded graph-Poisson spread of energy
  seam_openness.py      seam_openness: 1 relaxed .. 0 at the seal point, per seam
  leak.py               leak: each volume leaks energy along open seams, by pull
  first_mesh.py         first_mesh: the skeleton, the one Delaunay cut of a jittered lattice
  cell_complex.py       CellComplex: corner keys, own tetrahedra, triangles, cell kinds
  joint_cells.py        joint_cells: the cells cut from the skeleton, joints cells of their own
  joint_offsets.py      joint_offsets: how far each energy volume's faces stand back
  place_corners.py      place_corners: corner positions from the skeleton and offsets
  woven_corners.py      woven_corners: corners kept where they fit, placed afresh elsewhere
  keyed_corners.py      keyed_corners: where each corner was before, by its key
  relax_skeleton.py     relax_skeleton: the skeleton's corners move so tetrahedra follow targets
  cell_sizes.py         cell_sizes: every cell's measured volume from its triangles
  tet_faces.py          tet_faces: each skeleton triangle once, outward from its first cell
  cell_volumes.py       cell_volumes: each tetrahedron's measured volume
  face_areas.py         face_areas: each triangle's area
  cell_targets.py       cell_targets: rest for volumes, thickness x area for joint cells
  relax_corners.py      relax_corners: every corner moves so cells follow their targets
  face_joints.py        face_joints: one joint per skeleton triangle between two tetrahedra
  joint_carry.py        joint_carry: thickness and rate carried to the new joints
  mesh_seams.py         mesh_seams: seam cells two or more joint cells touch
  collapse_edge.py      collapse_edge: one edge drawn into a corner, its tetrahedra fall in
  split_cell.py         split_cell: the 1-4 move on the skeleton, a new corner at the centroid
  drop_loose_corners.py drop_loose_corners: skeleton corners no tetrahedron uses go
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
from physics.cells.seam_openness import seam_openness
from physics.cells.leak import leak
from physics.cells.tet_faces import tet_faces
from physics.cells.cell_volumes import cell_volumes
from physics.cells.face_areas import face_areas
from physics.cells.cell_sizes import cell_sizes
from physics.cells.cell_targets import cell_targets
from physics.cells.relax_corners import relax_corners
from physics.cells.first_mesh import first_mesh
from physics.cells.cell_complex import CellComplex
from physics.cells.joint_cells import joint_cells
from physics.cells.joint_offsets import joint_offsets
from physics.cells.place_corners import place_corners
from physics.cells.woven_corners import woven_corners
from physics.cells.keyed_corners import keyed_corners
from physics.cells.relax_skeleton import relax_skeleton
from physics.cells.face_joints import face_joints
from physics.cells.joint_carry import joint_carry
from physics.cells.mesh_seams import mesh_seams
from physics.cells.collapse_edge import collapse_edge
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
           "joint_push", "joint_balance", "joint_squeeze", "swing_joints", "seam_openness", "leak",
           "tet_faces", "cell_volumes", "face_areas", "cell_targets", "relax_corners",
           "first_mesh", "CellComplex", "joint_cells", "joint_offsets", "place_corners", "woven_corners",
           "keyed_corners", "relax_skeleton", "cell_sizes", "face_joints", "joint_carry", "mesh_seams", "collapse_edge", "split_cell",
           "drop_loose_corners", "Engine", "new_engine", "step", "lumps", "engine_stats",
           "mesh_message", "frame_arrays"]
