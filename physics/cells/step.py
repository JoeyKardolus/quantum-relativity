"""One step of the cell rules, all always on, using no shapes and no coordinates.

Contract: `step(grid, settings)` returns a new Grid, possibly with fewer energy
volumes (joint collapse), merged_into mapping every old volume to its new index.
Pull and push are read from the grid at the start of the step:
  1. pull       p = pull_potential(grid): (L + GROUND I) p = E (pull_potential.py).
  2. squeeze    joint (a, b): s_ab = attraction * (p_a + p_b) / 2, how hard the two
                volumes are drawn together through it (stored as Grid.squeeze).
  3. push       joint (a, b): u_ab = min(joint_push * s_ab, push_limit).
  4. volume     V'_a = V_a * (1 + drift) * exp(mean over a's joints of (u - s)):
                pull shrinks the volumes on both sides of a joint, push grows them
                back, drift grows every volume alike.
  5. energy-volume  the rule length = VOLUME_SCALE / E (length = V ** (1/3)) is
                restored, the mismatch m = ln(E * V'^(1/3) / VOLUME_SCALE) split
                evenly in log length: E *= exp(-m / 2), V = rest_volume(E). So a
                squeezed volume gains energy and a grown one loses it (redshift).
  6. flow       E = energy_flow(grid, E, energy_flow), V = rest_volume(E): total
                energy kept exactly.
  7. collapse   every joint asked for more push than it can give,
                joint_push * s_ab > push_limit (so s_ab > push_limit / joint_push),
                gives way: its two volumes fall into each other (joint_collapse.py),
                keeping all energy.
Deterministic: the same grid and settings give the same result.
"""
from __future__ import annotations

import numpy as np

from physics.cells.energy_flow import energy_flow
from physics.cells.grid import Grid
from physics.cells.joint_collapse import joint_collapse
from physics.cells.pull_potential import pull_potential
from physics.cells.rest_volume import VOLUME_SCALE, rest_volume
from physics.cells.settings import CellSettings


def step(grid: Grid, settings: CellSettings) -> Grid:
    """Apply pull, push, drift, the energy-volume rule, flow and collapse once."""
    potential = pull_potential(grid)
    heads, tails = grid.joints[:, 0], grid.joints[:, 1]
    squeeze = settings.attraction * (potential[heads] + potential[tails]) / 2
    pushed = np.minimum(settings.joint_push * squeeze, settings.push_limit)
    degree = np.maximum(np.bincount(grid.joints.ravel(), minlength=grid.cell_count), 1)
    net = pushed - squeeze
    mean_change = (np.bincount(heads, net, grid.cell_count)
                   + np.bincount(tails, net, grid.cell_count)) / degree
    volume = grid.volume * (1 + settings.drift) * np.exp(mean_change)
    mismatch = np.log(grid.energy * np.cbrt(volume) / VOLUME_SCALE)
    energy = grid.energy * np.exp(-mismatch / 2)
    flowed = Grid(cell_count=grid.cell_count, joints=grid.joints, energy=energy,
                  volume=rest_volume(energy), squeeze=squeeze)
    energy = energy_flow(flowed, energy, settings.energy_flow)
    moved = Grid(cell_count=grid.cell_count, joints=grid.joints, energy=energy,
                 volume=rest_volume(energy), squeeze=squeeze)
    return joint_collapse(moved, settings.joint_push * squeeze > settings.push_limit)
