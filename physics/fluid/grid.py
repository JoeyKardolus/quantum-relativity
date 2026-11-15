"""Grid: the foam as an energy fluid on a fixed cube of side**3 boxes on the GPU.

Contract: a Grid holds Warp arrays on its device (cuda unless given), each indexed
[x, y, z] with 0 <= x, y, z < side:
  a_energy    int64, the pulling (A) energy in each box, counted in QUANTUM units
  b_energy    int64, the pushing (B) energy in each box, in QUANTUM units
  a_momentum  vec3, A-energy times A's flow velocity (energy * boxes / time)
  b_momentum  vec3, the same for B
  potential   float32, the pull potential (pull_potential.py), kept to warm-start the next solve
Energy is held as whole quanta so that every flux between two boxes is one integer
taken from one and given to the other: total energy is exact, not merely close.
QUANTUM = 2**-32 energy; a box holding more than about 2e9 energy would overflow.
The contract's single "momentum (3 components)" is two, one per kind: A passing
through B (friction, the seal) is A moving relative to B, which one momentum field
cannot hold.
Start references, set by `Grid.from_host` and never changed by a step:
  start_total     total energy in quanta (Python int)
  start_a_mean    mean A-energy per box over the boxes holding A at the start
  start_occupied  boxes holding energy above settings.occupied_floor * start_a_mean
                  (default floor) at the start
steps counts step() calls since the start; substeps the substeps they took.
`Grid.from_host(a, b, a_velocity, b_velocity, device)` builds one from numpy arrays:
a, b float energy per box (>= 0, shape (side, side, side)), velocities (side, side,
side, 3) or None for rest. Raises ValueError for a non-cube or negative energy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from physics.fluid.settings import Settings

QUANTUM = 2.0 ** -32


@dataclass(eq=False)
class Grid:
    """A-energy, B-energy, their momenta and the pull potential per box, plus the start references."""

    side: int
    a_energy: wp.array
    b_energy: wp.array
    a_momentum: wp.array
    b_momentum: wp.array
    potential: wp.array
    start_total: int
    start_a_mean: float
    start_occupied: int
    steps: int = 0
    substeps: int = 0

    @classmethod
    def from_host(cls, a: np.ndarray, b: np.ndarray, a_velocity: np.ndarray | None = None,
                  b_velocity: np.ndarray | None = None, device: str = "cuda") -> "Grid":
        """A grid from energy per box and flow velocity per box, quantised and uploaded."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        side = a.shape[0]
        if a.shape != (side, side, side) or b.shape != a.shape:
            raise ValueError(f"energies must be one cube, got {a.shape} and {b.shape}")
        if np.any(a < 0) or np.any(b < 0) or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("energy per box must be finite and >= 0")
        a_quanta = np.rint(a / QUANTUM).astype(np.int64)
        b_quanta = np.rint(b / QUANTUM).astype(np.int64)
        a_float, b_float = a_quanta * QUANTUM, b_quanta * QUANTUM
        holding = a_float > 0
        start_a_mean = float(a_float[holding].mean()) if holding.any() else 1.0
        floor = Settings.all_on().occupied_floor * start_a_mean
        return cls(side=side,
                   a_energy=wp.array(a_quanta, dtype=wp.int64, device=device),
                   b_energy=wp.array(b_quanta, dtype=wp.int64, device=device),
                   a_momentum=wp.array(_momentum(a_float, a_velocity), dtype=wp.vec3, device=device),
                   b_momentum=wp.array(_momentum(b_float, b_velocity), dtype=wp.vec3, device=device),
                   potential=wp.zeros((side, side, side), dtype=wp.float32, device=device),
                   start_total=int(a_quanta.sum()) + int(b_quanta.sum()),
                   start_a_mean=start_a_mean,
                   start_occupied=int(((a_float + b_float) > floor).sum()))


def _momentum(energy: np.ndarray, velocity: np.ndarray | None) -> np.ndarray:
    if velocity is None:
        return np.zeros(energy.shape + (3,), dtype=np.float32)
    return (energy[..., None] * np.asarray(velocity, dtype=np.float64)).astype(np.float32)
