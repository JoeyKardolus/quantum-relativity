"""Cells layer: the woven grid of energy volumes and joints, and the rules that step it.

Module map:
  grid.py             Grid: per energy volume its energy, volume and charge, the joints
                      between neighbours with each joint's volume; no shapes, no coordinates
  rest_volume.py      rest_volume: the energy-volume rule, volume = (VOLUME_SCALE / E) ** 3
  joint_rest.py       joint_rest: a joint's rest volume, a share of its neighbours' mean volume
  build_grid.py       build_grid: periodic tetrahedra as scaffolding, then dropped
  settings.py         CellSettings: rule strengths, all_on() defaults
  pull_potential.py   pull_potential: grounded graph-Poisson spread of energy
  leak.py             leak: each volume leaks energy to its neighbours by their pull
  share_space.py      share_space: an energy volume hands space to its joints by their size
  move_space.py       move_space: pull, push, energy-volume rule, leak; no collapse
  joint_collapse.py   joint_collapse: joints that gave way, volumes fall into each other
  joint_squeeze.py    joint_squeeze: 0..1 how close each joint is to its limit, for drawing
  step.py             step: move_space, then collapse of joints below their limit
  grid_stats.py       grid_stats: cells, collapsed, size, joint share, packing spread, lumps
"""
from physics.cells.grid import Grid
from physics.cells.rest_volume import VOLUME_SCALE, rest_volume
from physics.cells.joint_rest import joint_rest
from physics.cells.settings import CellSettings
from physics.cells.build_grid import build_grid
from physics.cells.pull_potential import pull_potential
from physics.cells.leak import leak
from physics.cells.share_space import share_space
from physics.cells.move_space import move_space
from physics.cells.joint_collapse import joint_collapse
from physics.cells.joint_squeeze import joint_squeeze
from physics.cells.step import step
from physics.cells.grid_stats import grid_stats

__all__ = ["Grid", "VOLUME_SCALE", "rest_volume", "joint_rest", "CellSettings", "build_grid",
           "pull_potential", "leak", "share_space", "move_space", "joint_collapse",
           "joint_squeeze", "step", "grid_stats"]
