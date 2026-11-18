"""Search plan: the sizes of a search, fixed before it starts.

Contract: `SearchPlan` is a frozen record.
  cells, steps, every   lattice places a side, steps a short run, steps between samples
  workers               universes side by side on the one GPU, one process each
  seconds               wall time a short run may take before it is stopped (None: no cap)
  default_seeds         the seeds the default settings are run on
  random_points         points of the Latin hypercube; random_run runs only the first so
                        many of them (None: all)
  refine_rounds, refine_best, refine_count   rounds; best points stepped round; steps a point
  hold_top, hold_steps, hold_seeds   how many of the best calm points are run long, how
                        long, and the further seeds a holder is run on
  sweep_at, sweep_seeds unit-cube places one setting at a time is moved to, and on which seeds
  lattice_seed          the seed of the points
  device                the Warp device the universes run on
`long_every()` is the sample spacing of a long run (as many samples as a short one) and
`long_seconds()` its wall-time cap, both scaled by hold_steps over steps.
Raises ValueError when cells, steps, every or workers is below 1.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchPlan:
    """The sizes of one search."""
    cells: int = 24
    steps: int = 5000
    every: int = 100
    workers: int = 12
    seconds: float | None = 240.0
    default_seeds: tuple = (1, 2, 3, 4, 5, 6)
    random_points: int = 3000
    random_run: int | None = None
    refine_rounds: int = 3
    refine_best: int = 12
    refine_count: int = 16
    hold_top: int = 100
    hold_steps: int = 20000
    hold_seeds: tuple = (2, 3)
    sweep_at: tuple = (0.1, 0.3, 0.5, 0.7, 0.9)
    sweep_seeds: tuple = (1, 2)
    lattice_seed: int = 20261118
    device: str = "cuda"

    def __post_init__(self) -> None:
        for name in ("cells", "steps", "every", "workers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")

    def long_every(self) -> int:
        """Steps between samples of a long run."""
        return max(self.every * self.hold_steps // self.steps, 1)

    def long_seconds(self) -> float | None:
        """The wall-time cap of a long run."""
        return None if self.seconds is None else self.seconds * self.hold_steps / self.steps
