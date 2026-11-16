"""State: the engine's boxes on the GPU, one signed whole number and one phase per Planck box, universes side by side.

Contract: `State.from_host(quanta, phase, seed=1, device=None, lattice=None)` copies signed
quanta and phases (radians) and the forced cell lattice to the GPU and returns the state
step() advances. `quanta` is a cube
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
  marks      uint8, per step which rule changed the box: 2 an owner change, 1 a collapse
             (step.py zeroes it; collapse.py also uses it for its two passes)
  tallies    int64 (universes, 4): per universe the quanta spread by drift, joint boxes
             collapsed, owner changes, minus quanta turned plus by collapse
  substeps   int32 (universes,): how many fluid substeps each universe takes this step
             (substeps.py; 1 until a step sets it), read by on_site, hop and leak
  capped     int64 numpy (universes,), on the host: steps that needed more substeps
             than settings.max_substeps and took only that many
  owner      int32, the cell of the forced cell lattice the box belongs to (an
             extended-lattice id over the batch's extent, cell_place.py; ring cells
             are empty space)
  target     int32, the cell whose moving region covers the box's centre this step
             (box_targets.py); the box moves to it only once its quanta flowed out
The forced cell lattice (owner, 2026-11-16): `lattice` is None, or a dict of host
arrays: "cells" (nx, ny, nz) real cells per axis or "cells_per_side" n for a cube,
"owner" (one box array, or one per universe, ids over (nx + 2, ny + 2, nz + 2)) and
"corners" (float (nx + 3, ny + 3, nz + 3, 3), or one per universe): the lattice
vertices' positions in box units. A cell's kind is fixed by its lattice index
(cell_kind.py), so every box's sign must match its owner's kind (ValueError otherwise).
`extent` is the batch's extended cells per axis (nx + 2, ...), the storage every
universe's ids and corners use; `cells` (int numpy (universes, 3), on the host) and
`counts` (vec3i per universe, on the device) are each universe's own real cells per
axis. A volume split (volume_split.py) inserts lattice planes in one universe: its
cells grow along one axis, the extent grows when it must, and a universe with fewer
cells than the extent leaves the indices past its ring unused (ring, no volume). Per
lattice:
  corners      vec3 (universes * (extent[0] + 1), extent[1] + 1, extent[2] + 1), moved
             every step (move_corners.py)
  cell_quanta  int64 (universes, cells of the extent): each cell's whole quanta
             ignoring sign, counted at the start of every step (move_corners.py)
  cell_size    float32 (universes, cells of the extent): each cell's volume in boxes
             from its corners, measured with the count (move_corners.py; drain.py fills
             boxes up to the cell's packing quanta / size)
  splits, torn int64 numpy (universes,), on the host: volume splits so far and the plus
             quanta their tears turned minus (volume_split.py)
`has_lattice` is False without a lattice: then every box belongs to cell 0, no corner
moves, no box changes owner and nothing leaks (hand-built starts for single rules: a
pull, a wave, a whirl).
28 bytes and one uint8 per box. `seed` is an int (universe u gets seed + u) or one int
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

from physics.engine.lattice_kinds import lattice_kinds


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
    substeps: object = None
    capped: object = None
    extent: tuple = (1, 1, 1)
    cells: object = None
    counts: object = None
    splits: object = None
    torn: object = None
    owner: object = None
    target: object = None
    corners: object = None
    cell_quanta: object = None
    cell_size: object = None

    @property
    def has_lattice(self) -> bool:
        """True when the boxes sit in a forced cell lattice."""
        return self.cells is not None and bool(self.cells.any())

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
    def from_host(cls, quanta: np.ndarray, phase: np.ndarray, seed=1, device: str | None = None,
                  lattice: dict | None = None) -> "State":
        """Copy signed quanta, phases and the cell lattice to the GPU; pull, clocks and tallies start at 0."""
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
        counts, owner, corners = _lattice(lattice, quanta)
        extent = tuple(int(count) + 2 for count in counts) if any(counts) else (1, 1, 1)
        cell_count = extent[0] * extent[1] * extent[2]
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
                   start=tuple(_start(cube) for cube in quanta),
                   substeps=wp.array(np.ones(universes, dtype=np.int32), dtype=wp.int32, device=device),
                   capped=np.zeros(universes, dtype=np.int64), extent=extent,
                   cells=np.tile(np.asarray(counts, dtype=np.int64), (universes, 1)),
                   counts=wp.array(np.tile(np.asarray(counts, dtype=np.int32), (universes, 1)), dtype=wp.vec3i,
                                   device=device),
                   splits=np.zeros(universes, dtype=np.int64), torn=np.zeros(universes, dtype=np.int64),
                   owner=wp.array(owner.reshape(stacked), dtype=wp.int32, device=device),
                   target=wp.array(owner.reshape(stacked), dtype=wp.int32, device=device),
                   corners=wp.array(corners.reshape((-1,) + corners.shape[2:]), dtype=wp.vec3, device=device),
                   cell_quanta=wp.zeros((universes, cell_count), dtype=wp.int64, device=device),
                   cell_size=wp.zeros((universes, cell_count), dtype=wp.float32, device=device))


def _lattice(lattice: dict | None, quanta: np.ndarray) -> tuple:
    """(real cells per axis, owner per box, corners per universe) as checked host arrays; cell 0 without a lattice."""
    universes = quanta.shape[0]
    if lattice is None:
        return (0, 0, 0), np.zeros(quanta.shape, dtype=np.int32), np.zeros((universes, 2, 2, 2, 3), dtype=np.float32)
    counts = lattice.get("cells", lattice.get("cells_per_side"))
    counts = (int(counts),) * 3 if np.ndim(counts) == 0 else tuple(int(count) for count in counts)
    kinds = lattice_kinds(counts).ravel()
    vertices = tuple(count + 3 for count in counts)
    owner = np.broadcast_to(np.asarray(lattice["owner"], dtype=np.int32), quanta.shape)
    corners = np.broadcast_to(np.asarray(lattice["corners"], dtype=np.float32), (universes,) + vertices + (3,))
    if owner.min() < 0 or owner.max() >= kinds.size:
        raise ValueError(f"an owner lies outside the {kinds.size} cells of the lattice")
    if not np.array_equal(kinds[owner], np.sign(quanta)):
        raise ValueError("every box's sign must match its owner's kind (plus volume, minus joint, 0 ring)")
    return counts, np.ascontiguousarray(owner), np.ascontiguousarray(corners)


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
