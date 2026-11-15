"""Universe: the foam as cells of fixed Planck boxes on the GPU (the cellular Potts picture).

Contract: a Universe holds Warp arrays on its device (cuda unless given).
Per Planck box, indexed [x, y, z], 0 <= x, y, z < side:
  owner       int32, the cell the box belongs to, or -1 for an empty box
  energy      int64, the energy in the box in QUANTUM units (2**-32 energy; a box over ~2e9 overflows); its
              kind, pulling (A) or pushing (B), is its owner's kind; 0 in an empty box
  momentum    vec3, energy times flow velocity (energy * boxes / time)
  heat        float32, the undirected jiggling friction made of directed flow (energy
              units, kinetic energy lost at joint crossings and closed faces); it moves
              with the energy and is bookkeeping: total energy is the quanta alone
  potential   float32, the pull potential, kept to warm-start the next solve
  box_level   int32, the box's time level this step (time_levels.py)
  seam        int32, 1 for a joint's box on a seam this step (seam_flags.py)
  energy_next, momentum_next, heat_next: the flow's write buffers
Per cell, indexed by cell id 0 <= id < cell_count:
  kind        int32, 0 = energy volume (A, pulls), 1 = joint (B, pushes)
  alive       int32, 1 while the cell exists; 0 after it vanished in a joint collapse
              or fell into another energy volume
  labels      float32, how many start cells it stands for (1 at the start; summed when
              energy volumes fall together, shared out by box count in a split); its
              wanted size scales with it
  cell_energy int64 quanta, cell_boxes int32, cell_potential float32 (summed over its
              boxes), cell_place int64 (cell_count * 3, the sum of its boxes' indices: its
              centre is cell_place / cell_boxes), cell_heat float32, set by cell_sums.py;
              cell_want, cell_compression (want / boxes), cell_clock, cell_potential_mean
              float32, derived there
  cell_aim    float32, the size the cell's box moves aim at: its want, or for a joint
              its swing's next size (joint_swing.py); previous_boxes float32 its box
              count one step ago (the swing's memory)
  cell_speed  float32, cell_level int32: its fastest flow and its time level (time_levels.py)
  gained, lost int32, the boxes it gained and lost in the current sweep of moves
  neighbour, faces int32, cell_count * slots: the contact table (contact_table.py),
              slots its row width (SLOTS at the start, doubled when a row overflows);
              links int32, cell_count * link_slots: every cell each cell touches (link_table.py)
Active region: low, high (host int triples), the occupied boxes' bounds widened by
PAD and clipped to the grid (active_region.py); box kernels run only there.
Start references, set by `Universe.from_host`, never changed by a step:
  start_total quanta; start_occupied boxes; start_cells (A) and start_joints (B);
  rest the mean energy per occupied box (both kinds, the clocks' and the pressure's
  unit), rest_a the mean per A box (the lumps' unit), cell_rest the mean energy per
  start cell (both kinds: one size law); base_boxes the start cell's box count k**3.
steps counts step() calls, substeps the finest fluid substeps, box_updates the box
updates they made (each box only at its own level), collapses the joints that
collapsed, splits the volumes that tore, last_moves the boxes that changed owner in
the last step; corrected is True once the start's size correction ran.
`Universe.from_host(owner, energy, kind, boxes_per_cell_side, seed, device)` builds one at rest from numpy arrays: owner (side, side, side) ints, energy
(side, side, side) float >= 0 (quantised here; must be 0 in empty boxes), kind per
cell. Raises ValueError for a non-cube, negative or misplaced energy.
`universe.add_cells(kinds)` appends cells (alive, labels 0, zero sums) and returns their ids.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from physics.potts.contact_table import contact_table
from physics.potts.link_table import link_table

QUANTUM = 2.0 ** -32
SLOTS = 8
ENERGY_VOLUME = 0
JOINT = 1
PAD = 4

_CELL_FIELDS = {"kind": (wp.int32, 0), "alive": (wp.int32, 1), "labels": (wp.float32, 0.0),
                "cell_energy": (wp.int64, 0), "cell_boxes": (wp.int32, 0), "cell_potential": (wp.float32, 0.0),
                "cell_heat": (wp.float32, 0.0), "cell_want": (wp.float32, 0.0), "cell_compression": (wp.float32, 0.0),
                "cell_clock": (wp.float32, 1.0), "cell_potential_mean": (wp.float32, 0.0),
                "cell_aim": (wp.float32, 0.0), "previous_boxes": (wp.float32, 0.0), "cell_speed": (wp.float32, 0.0),
                "cell_level": (wp.int32, 0), "gained": (wp.int32, 0), "lost": (wp.int32, 0)}


@dataclass(eq=False)
class Universe:
    """Boxes (owner, energy, momentum, heat, potential), cells (sums, wants, clocks, swing), tables, references."""

    side: int
    cell_count: int
    boxes_per_cell_side: int
    seed: int
    owner: wp.array
    energy: wp.array
    momentum: wp.array
    heat: wp.array
    potential: wp.array
    box_level: wp.array
    seam: wp.array
    energy_next: wp.array
    momentum_next: wp.array
    heat_next: wp.array
    kind: wp.array
    alive: wp.array
    labels: wp.array
    cell_energy: wp.array
    cell_boxes: wp.array
    cell_potential: wp.array
    cell_heat: wp.array
    cell_place: wp.array
    cell_want: wp.array
    cell_compression: wp.array
    cell_clock: wp.array
    cell_potential_mean: wp.array
    cell_aim: wp.array
    previous_boxes: wp.array
    cell_speed: wp.array
    cell_level: wp.array
    gained: wp.array
    lost: wp.array
    neighbour: wp.array
    faces: wp.array
    slots: int
    links: wp.array
    link_slots: int
    low: tuple
    high: tuple
    start_total: int
    start_occupied: int
    start_cells: int
    start_joints: int
    rest: float
    rest_a: float
    cell_rest: float
    base_boxes: int
    steps: int = 0
    substeps: int = 0
    box_updates: int = 0
    collapses: int = 0
    splits: int = 0
    last_moves: int = 0
    corrected: bool = False
    scratch: dict = field(default_factory=dict)

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
        boxes = np.bincount(owner[occupied], minlength=cell_count).astype(np.float32)
        neighbour, faces, slots = contact_table(owner, cell_count, SLOTS)
        links, link_slots = link_table(owner, cell_count, 2 * SLOTS)
        places = np.argwhere(occupied)
        low = tuple(int(max(0, value - PAD)) for value in places.min(axis=0)) if places.size else (0, 0, 0)
        high = tuple(int(min(side, value + 1 + PAD)) for value in places.max(axis=0)) if places.size else (side,) * 3
        zeros_box = lambda dtype: wp.zeros((side,) * 3, dtype=dtype, device=device)
        cells = {name: wp.full(cell_count, value, dtype=dtype, device=device)
                 for name, (dtype, value) in _CELL_FIELDS.items()}
        cells["kind"] = wp.array(kind, dtype=wp.int32, device=device)
        cells["labels"] = wp.full(cell_count, 1.0, dtype=wp.float32, device=device)
        cells["previous_boxes"] = wp.array(boxes, dtype=wp.float32, device=device)
        return cls(side=side, cell_count=cell_count, boxes_per_cell_side=int(boxes_per_cell_side), seed=int(seed),
                   owner=wp.array(owner, dtype=wp.int32, device=device),
                   energy=wp.array(quanta, dtype=wp.int64, device=device),
                   momentum=zeros_box(wp.vec3), heat=zeros_box(wp.float32), potential=zeros_box(wp.float32),
                   box_level=zeros_box(wp.int32), seam=zeros_box(wp.int32), energy_next=zeros_box(wp.int64),
                   momentum_next=zeros_box(wp.vec3), heat_next=zeros_box(wp.float32),
                   cell_place=wp.zeros(cell_count * 3, dtype=wp.int64, device=device),
                   neighbour=wp.array(neighbour, dtype=wp.int32, device=device),
                   faces=wp.array(faces, dtype=wp.int32, device=device), slots=slots,
                   links=wp.array(links, dtype=wp.int32, device=device), link_slots=link_slots,
                   low=low, high=high,
                   start_total=int(quanta.sum()), start_occupied=int(occupied.sum()),
                   start_cells=int(np.sum(kind == ENERGY_VOLUME)), start_joints=int(np.sum(kind == JOINT)),
                   rest=_mean(quanta[occupied] * QUANTUM),
                   rest_a=_mean(quanta[box_kind == ENERGY_VOLUME] * QUANTUM),
                   cell_rest=_mean(per_cell),
                   base_boxes=int(boxes_per_cell_side) ** 3, **cells)

    def add_cells(self, kinds: list[int]) -> list[int]:
        """Append living cells of the given kinds (labels 0, zero sums); return their ids."""
        first = self.cell_count
        count = len(kinds)
        device = self.owner.device
        for name, (dtype, value) in _CELL_FIELDS.items():
            old = getattr(self, name)
            grown = wp.full(first + count, value, dtype=dtype, device=device)
            wp.copy(grown, old, count=first)
            setattr(self, name, grown)
        self.kind.assign(np.concatenate([self.kind.numpy()[:first], np.asarray(kinds, dtype=np.int32)]))
        place = wp.zeros((first + count) * 3, dtype=wp.int64, device=device)
        wp.copy(place, self.cell_place, count=first * 3)
        self.cell_place = place
        self.cell_count = first + count
        return list(range(first, first + count))


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size and values.mean() > 0 else 1.0
