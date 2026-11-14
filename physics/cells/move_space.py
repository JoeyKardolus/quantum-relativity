"""Move space: pull, joint push and the leak, without joint collapse.

Contract: `move_space(grid, settings)` returns a new Grid with the same energy
volumes and joints (merged_into None, squeeze unchanged: apply_rules() sets both).
Pull, push and the leak's shares are read from the grid at the start. An energy
volume's size is set only by its energy (rule 4): neither pull nor push touches
it, and no rule here derives energy from a volume. Energy moves only by the leak.
  1. pull     p = pull_potential(grid); joint (a, b) is pulled by
              s = attraction * (p_a + p_b) / 2 and keeps J * exp(-s) of its volume.
              The space squeezed out is gone (gravity contracts space).
  2. push     the joint swells back toward its rest volume R (joint_rest.py, read
              from the energy volumes at the start) by g = joint_push * (R - J)
              (a joint above rest thins the same way). The swelling is new space,
              taken from no energy volume: a push moves no energy (rule 8).
  3. leak     E = leak(joints, seams, E, attraction * p / 2, openness, settings.leak)
              (leak.py), openness = seam_openness(grid) at the start: each volume
              leaks that share of its energy, times its seams' mean openness, along
              its open seams to its neighbours by how hard each pulls; a sealed seam
              passes nothing; total energy kept exactly (to rounding).
  4. size     V = rest_volume(E): volume follows energy. A drained volume grows,
              a fed one shrinks; the change is new space (drift), not taken from
              joints. rest_volume is convex in E, so concentrating energy grows space.
So S_new = sum rest_volume(E after leak) + sum (J e^-s + g). Charge is carried
unchanged. Deterministic.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.cells.grid import Grid
from physics.cells.joint_rest import joint_rest
from physics.cells.leak import leak
from physics.cells.pull_potential import pull_potential
from physics.cells.rest_volume import rest_volume
from physics.cells.seam_openness import seam_openness
from physics.cells.engine_settings import EngineSettings


def move_space(grid: Grid, settings: EngineSettings) -> Grid:
    """One round of pull, joint push and leak; energy volumes keep the size their energy gives."""
    exerted = settings.attraction * pull_potential(grid) / 2
    openness = seam_openness(grid, settings)
    heads, tails = grid.joints[:, 0], grid.joints[:, 1]
    pulled = grid.joint_volume * np.exp(-(exerted[heads] + exerted[tails]))
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    joint_volume = pulled + settings.joint_push * (rest - pulled)
    leaked = leak(grid.joints, grid.seams, grid.energy, exerted, openness, settings.leak)
    return replace(grid, energy=leaked, volume=rest_volume(leaked), merged_into=None,
                   joint_volume=joint_volume)
