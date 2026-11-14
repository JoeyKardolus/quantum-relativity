"""The cell rules for one step, all always on, on the neighbour graph only (no shapes, no positions).

Contract: `apply_rules(grid, settings)` returns a new Grid, possibly with fewer energy
volumes (joint collapse), merged_into mapping every old volume to its new index:
  1. move     moved = move_space(grid, settings): pull squeezes joints, joints push
              back taking space from their energy volumes, the energy-volume rule
              and the leak (move_space.py has the formulas).
  2. collapse a joint gives way only when its volume fell below its limit,
              J < push_limit * R, R = joint_rest(moved volume, joints, joint_rest);
              its two energy volumes fall into each other (joint_collapse.py),
              keeping all energy and all space. No other path joins two volumes.
  3. squeeze  set to joint_squeeze(result, settings): how hard each joint pushes;
              seam_open to seam_openness(result, settings).
Total space changes only by pull (shrinks it) and the leak (drift: volumes
it drains grow). Deterministic: the same grid and
settings give the same result.
"""
from __future__ import annotations

from dataclasses import replace

from physics.cells.grid import Grid
from physics.cells.joint_collapse import joint_collapse
from physics.cells.joint_rest import joint_rest
from physics.cells.joint_squeeze import joint_squeeze
from physics.cells.move_space import move_space
from physics.cells.seam_openness import seam_openness
from physics.cells.engine_settings import EngineSettings


def apply_rules(grid: Grid, settings: EngineSettings) -> Grid:
    """Apply move_space, then collapse every joint below its limit."""
    moved = move_space(grid, settings)
    limit = settings.push_limit * joint_rest(moved.volume, moved.joints, settings.joint_rest)
    collapsed = joint_collapse(moved, moved.joint_volume < limit)
    return replace(collapsed, squeeze=joint_squeeze(collapsed, settings),
                   seam_open=seam_openness(collapsed, settings))
