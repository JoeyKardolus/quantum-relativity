"""State: the engine's boxes on the GPU, one signed whole number and one phase per Planck box, universes side by side.

Contract: `State.from_host(quanta, phase, seed=1, device=None)` copies signed quanta and
phases (radians) to the GPU and returns the state step() advances. `quanta` is a cube
(side, side, side) for one universe, or (universes, side, side, side) for a batch of
universes stepped side by side in one GPU job (the harness, physics/bench); `phase`
has the same shape. The universes of a batch lie stacked along the first axis, so every
per-box array has shape (universes * side, side, side) and universe u holds first
indices u * side .. u * side + side - 1; no rule reads across from one universe to the
next (inside.py), so each is its own closed box of space. Per box:
  quanta     int32, the box's energy in whole quanta: plus = pulling energy (an energy
             volume's box), minus = pushing energy (a joint's box), 0 = no space. The
             amount ignoring sign is the only thing conserved, exactly (bible rule 4).
  phase      float32 in (-pi, pi], the quantum fluid's phase. The fluid's complex number
             is never stored: it is sqrt(|quanta|) * exp(i phase), so its size squared is
             the box's whole quanta by construction (hop.py moves quanta only in whole
             numbers and keeps this tie after every move).
  pull       float32, the pull field (energy per plus quantum), spread at light speed
  pull_rate  float32, its time derivative (leapfrog, pull_wave.py)
  clock      float32, the proper time this box has lived (local clocks, tick_clocks.py)
  marks      uint8 scratch for two-pass kernels (collapse.py)
  tallies    int64 (universes, 4): per universe the quanta spread by drift, joint boxes
             collapsed, claims made, spare
20 bytes and one uint8 per box. `seed` is an int (universe u gets seed + u) or one int
per universe; `seeds` keeps them and `keys` (int32, on the device) their hashes, which
seed each universe's whole-number rounding (hop.py): a universe with seed s steps the
same alone as side by side in a batch. `step` counts steps (the counter, not time).
`start` holds per universe the start's reference, a dict of exact ints: amount, plus,
minus, boxes (occupied) and plus_boxes, and "cells", the start's cell count when its
builder knows it (new_universe), else None. `ladder` is None or the detail ladder's store kept beside the boxes (ladder_sync.py).
`state.shape` is the per-box arrays' shape and `state.universe(u)` the slice of the
first axis that universe u holds. Raises ValueError on a non-cubic or mismatched start,
a seed count that is not one per universe, or a box of 2**30 quanta or more.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp


@dataclass
class State:
    """The engine's per-box arrays on the GPU, its universes' seeds and its step counter."""

    side: int
    universes: int
    quanta: wp.array3d
    phase: wp.array3d
    pull: wp.array3d
    pull_rate: wp.array3d
    clock: wp.array3d
    marks: wp.array3d
    tallies: wp.array
    seeds: tuple
    keys: wp.array
    start: tuple = ()
    step: int = 0
    ladder: object = None

    @property
    def shape(self) -> tuple:
        """The per-box arrays' shape: (universes * side, side, side)."""
        return (self.universes * self.side, self.side, self.side)

    def universe(self, index: int) -> slice:
        """The first-axis slice universe `index` holds."""
        if not 0 <= index < self.universes:
            raise ValueError(f"universe {index} is not one of {self.universes}")
        return slice(index * self.side, (index + 1) * self.side)

    @classmethod
    def from_host(cls, quanta: np.ndarray, phase: np.ndarray, seed=1, device: str | None = None) -> "State":
        """Copy signed quanta and phases to the GPU; pull, clocks and tallies start at 0."""
        quanta = np.asarray(quanta)
        phase = np.asarray(phase)
        if quanta.ndim == 3:
            quanta, phase = quanta[None], phase[None] if phase.ndim == 3 else phase
        if quanta.ndim != 4 or len(set(quanta.shape[1:])) != 1:
            raise ValueError(f"quanta must be a cube or a batch of cubes, got shape {quanta.shape}")
        if phase.shape != quanta.shape:
            raise ValueError(f"phase shape {phase.shape} differs from quanta shape {quanta.shape}")
        if np.abs(quanta.astype(np.int64)).max(initial=0) >= 2 ** 30:
            raise ValueError("a box may hold fewer than 2**30 quanta (int32 headroom for moves)")
        universes, side = int(quanta.shape[0]), int(quanta.shape[1])
        seeds = _seeds(seed, universes)
        stacked = (universes * side, side, side)
        wrapped = _wrapped(phase).reshape(stacked)
        keys = np.array([(value * 2654435761 + 12345) & 0x7FFFFFFF for value in seeds], dtype=np.int32)
        return cls(side=side, universes=universes,
                   quanta=wp.array(quanta.astype(np.int32).reshape(stacked), dtype=wp.int32, device=device),
                   phase=wp.array(wrapped, dtype=wp.float32, device=device),
                   pull=wp.zeros(stacked, dtype=wp.float32, device=device),
                   pull_rate=wp.zeros(stacked, dtype=wp.float32, device=device),
                   clock=wp.zeros(stacked, dtype=wp.float32, device=device),
                   marks=wp.zeros(stacked, dtype=wp.uint8, device=device),
                   tallies=wp.zeros((universes, 4), dtype=wp.int64, device=device),
                   seeds=seeds, keys=wp.array(keys, dtype=wp.int32, device=device),
                   start=tuple(_start(cube) for cube in quanta))


def _wrapped(phase: np.ndarray) -> np.ndarray:
    """Phases brought into [-pi, pi) as float32, a zero array copied without arithmetic (large starts)."""
    angles = np.asarray(phase, dtype=np.float32)
    if not angles.any():
        return angles.copy() if angles is phase else angles
    return (np.remainder(angles.astype(np.float64) + np.pi, 2.0 * np.pi) - np.pi).astype(np.float32)


def _seeds(seed, universes: int) -> tuple:
    """One int seed per universe: seed + u for an int, else the given sequence."""
    if np.ndim(seed) == 0:
        return tuple(int(seed) + index for index in range(universes))
    seeds = tuple(int(value) for value in seed)
    if len(seeds) != universes:
        raise ValueError(f"need one seed per universe ({universes}), got {len(seeds)}")
    return seeds


def _start(cube: np.ndarray) -> dict:
    """One universe's start reference: its amounts and occupied boxes."""
    plus = cube[cube > 0].astype(np.int64)
    minus = -cube[cube < 0].astype(np.int64)
    return {"amount": int(plus.sum() + minus.sum()), "plus": int(plus.sum()), "minus": int(minus.sum()),
            "boxes": int(plus.size + minus.size), "plus_boxes": int(plus.size), "cells": None}
