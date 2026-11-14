"""Move space: pull, push, the energy-volume rule and the leak, without joint collapse.

Contract: `move_space(grid, settings)` returns a new Grid with the same energy
volumes and joints (merged_into None, squeeze unchanged: apply_rules() sets both).
Total space S = sum V + sum J. Pull, push and the leak's shares are read from
the grid at the start:
  1. pull     p = pull_potential(grid); joint (a, b) is pulled by
              s = attraction * (p_a + p_b) / 2 and keeps J * exp(-s) of its volume.
              The space squeezed out is gone (gravity contracts space).
  2. push     the joint swells back toward its rest volume R (joint_rest.py) by
              g = joint_push * (R - J), at most min(V_a / d_a, V_b / d_b) (d = joints
              of a volume, so no energy volume gives up more than half its volume).
              The swelling takes g / 2 from each of its two energy volumes (a joint
              above rest gives back the same way): S is kept.
  3. energy-volume  length = VOLUME_SCALE / E is restored, the mismatch
              m = ln(E * V^(1/3) / VOLUME_SCALE) split evenly in log length:
              E *= exp(-m / 2). So a volume compressed by its joints' push gains
              energy (packing) and a grown one loses it (redshift).
  4. settle   V = rest_volume(E); each energy volume hands the space it gave up
              (or takes the space it gained) to its own joints in proportion to
              their volumes (share_space.py): S is kept.
  5. leak     E = leak(joints, E, attraction * p / 2, settings.leak) (leak.py):
              each volume leaks that share of its energy to its neighbours, by how
              hard each pulls (share_b = p_b / sum of p over the leaker's
              neighbours); total energy kept exactly. Then V = rest_volume(E):
              volume follows energy, and the change is new space (drift), not taken
              from joints. A drained volume grows, a fed one shrinks; rest_volume
              is convex in E, so concentrating energy grows S and evening it shrinks S.
So S_new = S - pulled_out + sum rest_volume(E after leak) - sum rest_volume(E before).
Charge is carried unchanged. Deterministic. Raises ValueError (share_space) when
an energy volume would take more space than its joints hold.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.cells.grid import Grid
from physics.cells.joint_rest import joint_rest
from physics.cells.leak import leak
from physics.cells.pull_potential import pull_potential
from physics.cells.rest_volume import VOLUME_SCALE, rest_volume
from physics.cells.engine_settings import EngineSettings
from physics.cells.share_space import share_space


def move_space(grid: Grid, settings: EngineSettings) -> Grid:
    """One round of pull, push, energy-volume rule and leak; joints never give way here."""
    exerted = settings.attraction * pull_potential(grid) / 2
    heads, tails = grid.joints[:, 0], grid.joints[:, 1]
    pulled = grid.joint_volume * np.exp(-(exerted[heads] + exerted[tails]))
    swell = _swell(grid, settings, pulled)
    count = grid.cell_count
    given = (np.bincount(heads, swell, count) + np.bincount(tails, swell, count)) / 2
    volume = grid.volume - given
    mismatch = np.log(grid.energy * np.cbrt(volume) / VOLUME_SCALE)
    energy = grid.energy * np.exp(-mismatch / 2)
    joint_volume = share_space(grid.joints, pulled + swell, volume - rest_volume(energy))
    leaked = leak(grid.joints, energy, exerted, settings.leak)
    return replace(grid, energy=leaked, volume=rest_volume(leaked), merged_into=None,
                   joint_volume=joint_volume)


def _swell(grid: Grid, settings: EngineSettings, pulled: np.ndarray) -> np.ndarray:
    """Push per joint toward its rest volume, capped so no energy volume gives up over half."""
    heads, tails = grid.joints[:, 0], grid.joints[:, 1]
    degree = np.maximum(np.bincount(grid.joints.ravel(), minlength=grid.cell_count), 1)
    cap = np.minimum(grid.volume[heads] / degree[heads], grid.volume[tails] / degree[tails])
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    return np.minimum(settings.joint_push * (rest - pulled), cap)
