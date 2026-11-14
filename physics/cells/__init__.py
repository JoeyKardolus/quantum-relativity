"""Cells layer: the woven grid of energy volumes and joints, and the rules that step it.

Module map:
  grid.py             Grid: per energy volume its energy and volume, the joints between
                      neighbours with each joint's squeeze; no shapes, no coordinates
  rest_volume.py      rest_volume: the energy-volume rule, volume = (VOLUME_SCALE / E) ** 3
  build_grid.py       build_grid: periodic tetrahedra as scaffolding, then dropped
  settings.py         CellSettings: rule strengths, all_on() defaults
  pull_potential.py   pull_potential: grounded graph-Poisson spread of energy
  energy_flow.py      energy_flow: conserved flow between neighbours through joints
  joint_collapse.py   joint_collapse: joints that gave way, volumes fall into each other
  step.py             step: pull, push, drift, energy-volume, flow, collapse
  balancing_drift.py  balancing_drift: drift that keeps even space the same size
  grid_stats.py       grid_stats: cells, collapsed, size, packing spread, lumps
"""
from physics.cells.grid import Grid
from physics.cells.rest_volume import VOLUME_SCALE, rest_volume
from physics.cells.build_grid import build_grid
from physics.cells.settings import CellSettings
from physics.cells.pull_potential import pull_potential
from physics.cells.energy_flow import energy_flow
from physics.cells.joint_collapse import joint_collapse
from physics.cells.step import step
from physics.cells.balancing_drift import balancing_drift
from physics.cells.grid_stats import grid_stats

__all__ = ["Grid", "VOLUME_SCALE", "rest_volume", "build_grid", "CellSettings",
           "pull_potential", "energy_flow", "joint_collapse", "step", "balancing_drift",
           "grid_stats"]
