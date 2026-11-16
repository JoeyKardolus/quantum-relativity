"""Ladder settings: how quiet a block must be, and how coarse it may get, before it leaves full detail.

Contract: `LadderSettings` is a frozen dataclass read by `switch_levels`. These are accuracy
settings of the storage, not physics: they decide how much error a COARSE block may carry,
never what the field does. Never tune them to make a physics result pass.
  flow_tol         a block holds flow when any face between two boxes with space (nonzero
                   quanta) touching it has a phase step above this (radians per box); a
                   block with flow stays FULL
  max_rel_error    COARSE only when every group's quanta step - 1 (the worst error per
                   box, in quanta) is at most this share of the group's smallest magnitude
  max_phase_error  COARSE only when every group's phase error bound (span / 15) is at most
                   this (radians)
  quiet_steps      a FULL block is demoted only after this many quiet measurements in a row
  chunk_blocks     blocks measured per batch (bounds scratch memory)
BLOCK (8) boxes per block edge and GROUP (4) boxes per shared-scale group edge are fixed:
the kernels' bit packing is written for them (8 groups of 64 boxes per block, 4-bit offsets).
"""
from __future__ import annotations

from dataclasses import dataclass

BLOCK = 8
GROUP = 4
BOXES = BLOCK ** 3


@dataclass(frozen=True)
class LadderSettings:
    """How quiet a block must be, and how much error COARSE may carry."""

    flow_tol: float = 0.05
    max_rel_error: float = 0.01
    max_phase_error: float = 0.02
    quiet_steps: int = 4
    chunk_blocks: int = 32768
