"""Cells layer: the engine. Energy volumes and joints, the rules that step them, and the
exact, gap-free cell shapes they take in a periodic box.

The rules and every measurement read only energies, sizes and the neighbour graph
(Grid). A working position and a power weight per energy volume are labels the
engine keeps only to compute the shapes (a periodic power tessellation); the
joints are exactly the shared faces of that tessellation.

Module map:
  engine.py           Engine: Grid + positions, weights, box side, tessellation, collapses
  engine_settings.py  EngineSettings: rule strengths, all_on() defaults
  new_engine.py       new_engine: seeded uneven energy, random seeds, settled shapes
  step.py             step: pure Engine -> Engine; rules, then seeds, box and shapes follow
  shapes.py           shapes: cells' faces with pull, joints' faces with push, for drawing
  engine_stats.py     engine_stats: cells, joints, collapses, size, spread, lumps
  grid.py             Grid: per energy volume its energy, volume and charge, the joints
                      between neighbours with each joint's volume; no shapes, no coordinates
  apply_rules.py      apply_rules: move_space, then collapse of joints below their limit
  rest_volume.py      rest_volume: the energy-volume rule, volume = (VOLUME_SCALE / E) ** 3
  joint_rest.py       joint_rest: a joint's rest volume, a share of its neighbours' mean volume
  pull_potential.py   pull_potential: grounded graph-Poisson spread of energy
  leak.py             leak: each volume leaks energy to its neighbours by their pull
  share_space.py      share_space: an energy volume hands space to its joints by their size
  move_space.py       move_space: pull, push, energy-volume rule, leak; no collapse
  joint_collapse.py   joint_collapse: joints that gave way, volumes fall into each other
  joint_squeeze.py    joint_squeeze: 0..1 how close each joint is to its limit (its push)
  tessellation.py     Tessellation: volumes, centroids, faces, neighbour pairs, as arrays
  tessellate.py       tessellate: periodic power tessellation of positions (pyvoro2)
  target_size.py      target_size: energy volume plus half its joints, per cell
  fit_weights.py      fit_weights: Newton step of the weights toward the target sizes
  move_seeds.py       move_seeds: seeds relax to centroids, drawn across squeezed joints
  reconcile_joints.py reconcile_joints: joints become the new faces, joint volumes kept
"""
from physics.cells.grid import Grid
from physics.cells.rest_volume import VOLUME_SCALE, rest_volume
from physics.cells.joint_rest import joint_rest
from physics.cells.engine_settings import EngineSettings
from physics.cells.pull_potential import pull_potential
from physics.cells.leak import leak
from physics.cells.share_space import share_space
from physics.cells.move_space import move_space
from physics.cells.joint_collapse import joint_collapse
from physics.cells.joint_squeeze import joint_squeeze
from physics.cells.apply_rules import apply_rules
from physics.cells.tessellation import Tessellation
from physics.cells.tessellate import tessellate
from physics.cells.target_size import target_size
from physics.cells.fit_weights import fit_weights
from physics.cells.move_seeds import move_seeds
from physics.cells.reconcile_joints import reconcile_joints
from physics.cells.engine import Engine
from physics.cells.new_engine import new_engine
from physics.cells.step import step
from physics.cells.shapes import shapes
from physics.cells.engine_stats import engine_stats

__all__ = ["Grid", "VOLUME_SCALE", "rest_volume", "joint_rest", "EngineSettings",
           "pull_potential", "leak", "share_space", "move_space", "joint_collapse",
           "joint_squeeze", "apply_rules", "Tessellation", "tessellate", "target_size",
           "fit_weights", "move_seeds", "reconcile_joints", "Engine", "new_engine", "step",
           "shapes", "engine_stats"]
