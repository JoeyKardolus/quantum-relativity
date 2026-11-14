"""Move space: pull, push, drift, the energy-volume rule and flow, without joint collapse.

Contract: `move_space(grid, settings)` returns a new Grid with the same energy
volumes and joints (merged_into None, squeeze unchanged: step() sets both).
Total space S = sum V + sum J. Pull and push are read from the grid at the start:
  1. pull     p = pull_potential(grid); joint (a, b) is pulled by
              s = attraction * (p_a + p_b) / 2 and keeps J * exp(-s) of its volume.
              The space squeezed out is gone: pull is the one rule that shrinks space
              (gravity contracts it), and drift balances it.
  2. push     the joint swells back toward its rest volume R (joint_rest.py) by
              g = joint_push * (R - J), at most min(V_a / d_a, V_b / d_b) (d = joints
              of a volume, so no energy volume gives up more than half its volume).
              The swelling takes g / 2 from each of its two energy volumes (a joint
              above rest gives back the same way): S is kept.
  3. drift    every energy volume and every joint grows by (1 + drift).
  4. energy-volume  length = VOLUME_SCALE / E is restored, the mismatch
              m = ln(E * V^(1/3) / VOLUME_SCALE) split evenly in log length:
              E *= exp(-m / 2). So a volume compressed by its joints' push gains
              energy (packing) and a grown one loses it (redshift).
  5. flow     E = energy_flow(grid, E, energy_flow): total energy kept exactly.
  6. settle   V = rest_volume(E); each energy volume hands the space it gave up
              (or takes the space it gained) to its own joints in proportion to
              their volumes (share_space.py): S is kept.
So S changes only by pull and drift: S_new = (S - pulled_out) * (1 + drift).
Charge is carried unchanged. Deterministic. Raises ValueError (share_space) when
an energy volume would take more space than its joints hold.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from physics.cells.energy_flow import energy_flow
from physics.cells.grid import Grid
from physics.cells.joint_rest import joint_rest
from physics.cells.pull_potential import pull_potential
from physics.cells.rest_volume import VOLUME_SCALE, rest_volume
from physics.cells.settings import CellSettings
from physics.cells.share_space import share_space


def move_space(grid: Grid, settings: CellSettings) -> Grid:
    """One round of pull, push, drift, energy-volume rule and flow; joints never give way here."""
    pulled = grid.joint_volume * np.exp(-_pull(grid, settings))
    swell = _swell(grid, settings, pulled)
    heads, tails = grid.joints[:, 0], grid.joints[:, 1]
    count = grid.cell_count
    given = (np.bincount(heads, swell, count) + np.bincount(tails, swell, count)) / 2
    grow = 1 + settings.drift
    volume = (grid.volume - given) * grow
    joint_volume = (pulled + swell) * grow
    mismatch = np.log(grid.energy * np.cbrt(volume) / VOLUME_SCALE)
    energy = energy_flow(grid, grid.energy * np.exp(-mismatch / 2), settings.energy_flow)
    settled = rest_volume(energy)
    return replace(grid, energy=energy, volume=settled, merged_into=None,
                   joint_volume=share_space(grid.joints, joint_volume, volume - settled))


def _pull(grid: Grid, settings: CellSettings) -> np.ndarray:
    potential = pull_potential(grid)
    return settings.attraction * (potential[grid.joints[:, 0]] + potential[grid.joints[:, 1]]) / 2


def _swell(grid: Grid, settings: CellSettings, pulled: np.ndarray) -> np.ndarray:
    """Push per joint toward its rest volume, capped so no energy volume gives up over half."""
    heads, tails = grid.joints[:, 0], grid.joints[:, 1]
    degree = np.maximum(np.bincount(grid.joints.ravel(), minlength=grid.cell_count), 1)
    cap = np.minimum(grid.volume[heads] / degree[heads], grid.volume[tails] / degree[tails])
    rest = joint_rest(grid.volume, grid.joints, settings.joint_rest)
    return np.minimum(settings.joint_push * (rest - pulled), cap)
