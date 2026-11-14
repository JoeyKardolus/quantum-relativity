"""One step of the cell rules, all always on, using no shapes and no coordinates.

Contract: `step(grid, settings)` returns a new Grid, possibly with fewer energy
volumes (joint collapse), merged_into mapping every old volume to its new index:
  1. move     moved = move_space(grid, settings): pull squeezes joints, joints push
              back taking space from their energy volumes, drift, the energy-volume
              rule and energy flow (move_space.py has the formulas).
  2. collapse a joint gives way only when its volume fell below its limit,
              J < push_limit * R, R = joint_rest(moved volume, joints, joint_rest);
              its two energy volumes fall into each other (joint_collapse.py),
              keeping all energy and all space. No other path joins two volumes.
  3. squeeze  set to joint_squeeze(result, settings), for drawing only.
Total space changes only by pull and drift. Deterministic: the same grid and
settings give the same result.
"""
from __future__ import annotations

from dataclasses import replace

from physics.cells.grid import Grid
from physics.cells.joint_collapse import joint_collapse
from physics.cells.joint_rest import joint_rest
from physics.cells.joint_squeeze import joint_squeeze
from physics.cells.move_space import move_space
from physics.cells.settings import CellSettings


def step(grid: Grid, settings: CellSettings) -> Grid:
    """Apply move_space, then collapse every joint below its limit."""
    moved = move_space(grid, settings)
    limit = settings.push_limit * joint_rest(moved.volume, moved.joints, settings.joint_rest)
    collapsed = joint_collapse(moved, moved.joint_volume < limit)
    return replace(collapsed, squeeze=joint_squeeze(collapsed, settings))
