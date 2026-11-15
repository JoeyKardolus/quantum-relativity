"""Universe: the foam as cells of fixed Planck boxes on the GPU (the cellular Potts picture).

Contract: a Universe holds Warp arrays on its device (cuda unless given).
Per Planck box, indexed [x, y, z], 0 <= x, y, z < side:
  owner       int32, the cell the box belongs to, or -1 for an empty box
  energy      int64, the energy in the box in QUANTUM units (2**-32 energy; a box over ~2e9 overflows); its
              kind, pulling (A) or pushing (B), is its owner's kind; 0 in an empty box
  momentum    vec3, energy times flow velocity (energy * boxes / time)
  potential   float32, the pull potential, kept to warm-start the next solve
  a_quanta    int64, scratch: the box's energy if its owner is A, else 0 (the pull's source)
Per cell, indexed by cell id 0 <= id < cell_count:
  kind        int32, 0 = energy volume (A, pulls), 1 = joint (B, pushes)
  alive       int32, 1 while the cell exists; 0 after it vanished in a joint collapse
              or fell into another energy volume
  labels      int32, how many start cells of its kind it stands for (1, more after
              energy volumes fall together); its wanted size scales with it
  cell_energy int64 quanta, cell_boxes int32, cell_potential float32 (summed over its
              boxes), set by cell_sums.py; cell_want, cell_compression (want / boxes),
              cell_clock, cell_potential_mean float32, derived there
  gained, lost int32, the boxes it gained and lost in the current move phase
  neighbour, faces int32, cell_count * slots: the contact table (contact_table.py),
              slots its row width (SLOTS at the start, doubled when a row overflows)
Start references, set by `Universe.from_host`, never changed by a step:
  start_total quanta; start_occupied boxes; start_cells (A) and start_joints (B);
  rest_a, rest_b the mean energy per box of each kind; cell_rest_a, cell_rest_b the
  mean energy per start cell of each kind; base_boxes the start cell's box count k**3.
steps counts step() calls, substeps the fluid substeps, collapses the joints that
collapsed, last_moves the boxes that changed owner in the last step.
`Universe.from_host(owner, energy, kind, boxes_per_cell_side, seed, device)` builds one at rest from numpy arrays: owner (side, side, side) ints, energy
(side, side, side) float >= 0 (quantised here; must be 0 in empty boxes), kind per
cell. Raises ValueError for a non-cube, negative or misplaced energy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from physics.potts.contact_table import contact_table

QUANTUM = 2.0 ** -32
SLOTS = 8
ENERGY_VOLUME = 0
JOINT = 1


@dataclass(eq=False)
class Universe:
    """Boxes (owner, energy, momentum, potential), cells (kind, sums, wants, clocks), contacts, start references."""

    side: int
    cell_count: int
    boxes_per_cell_side: int
    seed: int
    owner: wp.array
    energy: wp.array
    momentum: wp.array
    potential: wp.array
    a_quanta: wp.array
    energy_next: wp.array
    momentum_next: wp.array
    kind: wp.array
    alive: wp.array
    labels: wp.array
    cell_energy: wp.array
    cell_boxes: wp.array
    cell_potential: wp.array
    cell_want: wp.array
    cell_compression: wp.array
    cell_clock: wp.array
    cell_potential_mean: wp.array
    gained: wp.array
    lost: wp.array
    neighbour: wp.array
    faces: wp.array
    slots: int
    start_total: int
    start_occupied: int
    start_cells: int
    start_joints: int
    rest_a: float
    rest_b: float
    cell_rest_a: float
    cell_rest_b: float
    base_boxes: int
    steps: int = 0
    substeps: int = 0
    collapses: int = 0
    last_moves: int = 0

    @classmethod
    def from_host(cls, owner: np.ndarray, energy: np.ndarray, kind: np.ndarray, boxes_per_cell_side: int,
                  seed: int, device: str = "cuda") -> "Universe":
        """A universe at rest from box owners, box energies and cell kinds, quantised and uploaded."""
        owner = np.asarray(owner, dtype=np.int32)
        energy = np.asarray(energy, dtype=np.float64)
        kind = np.asarray(kind, dtype=np.int32)
        side = owner.shape[0]
        if owner.shape != (side, side, side) or energy.shape != owner.shape:
            raise ValueError(f"owner and energy must be one cube, got {owner.shape} and {energy.shape}")
        if not np.all(np.isfinite(energy)) or np.any(energy < 0):
            raise ValueError("energy per box must be finite and >= 0")
        if np.any(energy[owner < 0] > 0):
            raise ValueError("an empty box holds no energy")
        cell_count = int(kind.size)
        quanta = np.rint(energy / QUANTUM).astype(np.int64)
        occupied = owner >= 0
        box_kind = np.where(occupied, kind[np.maximum(owner, 0)], -1)
        per_cell = np.bincount(owner[occupied], weights=quanta[occupied] * QUANTUM, minlength=cell_count)
        neighbour, faces, slots = contact_table(owner, cell_count, SLOTS)
        zeros_box = lambda dtype: wp.zeros((side,) * 3, dtype=dtype, device=device)
        zeros_cell = lambda dtype: wp.zeros(cell_count, dtype=dtype, device=device)
        return cls(side=side, cell_count=cell_count, boxes_per_cell_side=int(boxes_per_cell_side), seed=int(seed),
                   owner=wp.array(owner, dtype=wp.int32, device=device),
                   energy=wp.array(quanta, dtype=wp.int64, device=device),
                   momentum=zeros_box(wp.vec3), potential=zeros_box(wp.float32), a_quanta=zeros_box(wp.int64),
                   energy_next=zeros_box(wp.int64), momentum_next=zeros_box(wp.vec3),
                   kind=wp.array(kind, dtype=wp.int32, device=device),
                   alive=wp.array(np.ones(cell_count, dtype=np.int32), dtype=wp.int32, device=device),
                   labels=wp.array(np.ones(cell_count, dtype=np.int32), dtype=wp.int32, device=device),
                   cell_energy=zeros_cell(wp.int64), cell_boxes=zeros_cell(wp.int32),
                   cell_potential=zeros_cell(wp.float32), cell_want=zeros_cell(wp.float32),
                   cell_compression=zeros_cell(wp.float32),
                   cell_clock=wp.array(np.ones(cell_count, dtype=np.float32), dtype=wp.float32, device=device),
                   cell_potential_mean=zeros_cell(wp.float32), gained=zeros_cell(wp.int32),
                   lost=zeros_cell(wp.int32),
                   neighbour=wp.array(neighbour, dtype=wp.int32, device=device),
                   faces=wp.array(faces, dtype=wp.int32, device=device), slots=slots,
                   start_total=int(quanta.sum()), start_occupied=int(occupied.sum()),
                   start_cells=int(np.sum(kind == ENERGY_VOLUME)), start_joints=int(np.sum(kind == JOINT)),
                   rest_a=_mean(quanta[box_kind == ENERGY_VOLUME] * QUANTUM),
                   rest_b=_mean(quanta[box_kind == JOINT] * QUANTUM),
                   cell_rest_a=_mean(per_cell[kind == ENERGY_VOLUME]),
                   cell_rest_b=_mean(per_cell[kind == JOINT]),
                   base_boxes=int(boxes_per_cell_side) ** 3)


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size and values.mean() > 0 else 1.0
