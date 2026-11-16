"""Quanta batch: many small universes of signed whole quanta side by side in one GPU job.

Contract: `QuantaBatch(quanta, phase, steps=0)` holds one state for `universe_count`
universes of side**3 Planck boxes each, stacked along a leading batch dimension:
  quanta  wp.array4d int32, shape (universe_count, side, side, side): each box's one
          signed whole number (bible rule 4: plus is pulling energy, an energy volume;
          minus is pushing energy, a joint; zero is no space)
  phase   wp.array4d float32, same shape: the energy fluid's phase per box, radians
  steps   host int, the steps taken so far; a step function adds 1 per step
Universe u is quanta[u] and phase[u]; no box of one universe neighbours a box of
another, so a step function's neighbour lookups stop at the universe's own walls
(index 0 and side - 1 on each axis) and universes never exchange anything. The batch
dimension, not a big grid with empty separators, was chosen because a separator is
space a growing universe could claim (zero means no space, and space grows into it),
and it wastes boxes. Raises ValueError when quanta and phase are not one 4D shape of
equal sides, of int32 and float32, on one device.
"""
from __future__ import annotations

from dataclasses import dataclass

import warp as wp


@dataclass
class QuantaBatch:
    """Signed int32 quanta and float32 phase per box, for many universes at once."""

    quanta: wp.array
    phase: wp.array
    steps: int = 0

    def __post_init__(self) -> None:
        shape = tuple(self.quanta.shape)
        if self.quanta.ndim != 4 or shape[1] != shape[2] or shape[2] != shape[3]:
            raise ValueError(f"quanta must be (universes, side, side, side), got {shape}")
        if tuple(self.phase.shape) != shape:
            raise ValueError(f"phase must match quanta's shape {shape}, got {tuple(self.phase.shape)}")
        if self.quanta.dtype != wp.int32 or self.phase.dtype != wp.float32:
            raise ValueError(f"quanta must be int32 and phase float32, got {self.quanta.dtype}, {self.phase.dtype}")
        if self.quanta.device != self.phase.device:
            raise ValueError(f"quanta and phase on different devices: {self.quanta.device}, {self.phase.device}")

    @property
    def universe_count(self) -> int:
        """How many universes the batch holds."""
        return int(self.quanta.shape[0])

    @property
    def side(self) -> int:
        """Boxes along each axis of one universe."""
        return int(self.quanta.shape[1])

    @property
    def boxes_per_universe(self) -> int:
        """Planck boxes in one universe, side**3."""
        return self.side ** 3
