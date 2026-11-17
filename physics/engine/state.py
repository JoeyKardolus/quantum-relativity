"""State: the grid of cells, their relations, and each cell's own block of compute boxes, on the GPU.

Contract: `State(side, capacity, seed=1, device="cuda")` allocates an empty state for a
lattice of `side` places per axis (even, at least 4; it wraps round, there is no edge and
no empty space) with room for `capacity` cell ids; new_universe.py fills it. The cells
are the grid (axiom 3, owner 2026-11-17): a perfect A B A B pattern, an energy volume
where all three lattice indices are even, a joint everywhere else. A cell has no
geometry: only its relations (its neighbours, by the 26 directions) and a length set by
its energy alone, `size_energy[kind] / quanta`. Planck boxes are not space: every cell id
owns one fixed block of BLOCK ** 3 = 216 compute boxes, row `cell` of `quanta` and
`phase`, for good; no box ever changes cell and an id is never used again.
Per box (GPU, shape (capacity, 6, 6, 6)):
  quanta      int32 whole quanta: plus in an energy volume, minus in a joint, each box of
              a living cell at least 1 in size; all 0 in an id that is not a cell
  phase       float32 the fluid's phase; its complex number is sqrt(|quanta|) e^{i phase}
Per cell id (GPU, read with `host(name)`):
  kind        int32 1 energy volume, -1 joint, 0 no cell (ENERGY_VOLUME, JOINT)
  cell_quanta int64 the cell's quanta ignoring sign, always the sum over its boxes
  length      float32 its size by the law, exactly float32(size_energy[kind] / quanta)
  packing     float32 (start length of its kind / length) ** 3
  rate        float32 its local clock rate (clock_rate.py); clock float32 the time it lived
  pull, pull_old  float32 the pull there (energy per plus quantum) now and a step ago
  jiggle      float32 an energy volume's own heat per quantum (jiggle.py)
  squeeze     float32 a joint's squeeze this step, pull across it over its push (squeeze.py)
  gain, loss  int64 quanta the cell took and gave this step (crossing; a joint's gain is
              what it made); crossed int64 quanta that crossed a joint this step
  flags       int32 1 a joint to collapse, 2 a volume to tear, found this step
  ways_n      int32 a joint: the ways it carries (volumes on its one side times volumes
              on its other); a volume: the ways it is an end of; side_n int32 a joint:
              the volumes on its first side
`books` (GPU int64): [0] quanta made by squeeze, [1] quanta crossed, since the start.
Relations. A cell whose neighbours are still the lattice's is pristine and stores
nothing: its 26 neighbours and, for a face joint (one odd index), its one way (the two
volumes it lies between) follow from its id (lattice_step.py). A cell changed by a tear
or a collapse keeps its own list: on the host `adj_over[cell] = (ids, codes)` (code =
(dx + 1) * 9 + (dy + 1) * 3 + dz + 1, the direction the neighbour lies in; one neighbour
may fill several directions and one direction hold several neighbours) and
`sides_over[joint] = (volumes on one side, volumes on the other)`: its ways are every
pair of one volume from each side (a pair of a volume with itself is no way), kept as
two lists so that a joint between two much-torn volumes stays small; relink() copies
them to the GPU pools (adj_at, adj_n, adj_pool, way_at, side_n, way_pool). `version`
counts tears and collapses.
Host numbers: side, lattice (side ** 3), count (ids used), capacity, step, seed, key,
size_energy {1: , -1: } the size law's constants, start_quanta {1: , -1: } the start's
mean quanta per kind, start_length {1: , -1: }, collapses, merges, tears, torn (plus
quanta turned minus by tears), taken_joint and taken_volumes (quanta a collapse took
off the joint and off its volumes), fell_into {vanished volume: the
volume it fell into}, start (totals at step 0), quantum (None or quantum_store.py).
Methods: neighbours(cell) -> (ids, codes); set_neighbours; sides(joint) -> (first side,
second side); set_sides (keeps every volume's count of ways, ways_h, in step);
ways(joint) -> [(a, b)]; joints_of(cell); block(cell) -> (quanta, phase) copies;
set_block; fill(cell, amount) stages a cell's quanta evenly; host(name); grow(); relink().
"""
from __future__ import annotations

import numpy as np
import warp as wp

BLOCK = 6
BOXES = BLOCK ** 3
ENERGY_VOLUME = 1
JOINT = -1
CENTRE = 13

_CELL_ARRAYS = {"kind": wp.int32, "cell_quanta": wp.int64, "length": wp.float32, "packing": wp.float32,
                "rate": wp.float32, "clock": wp.float32, "pull": wp.float32, "pull_old": wp.float32,
                "pull_new": wp.float32, "jiggle": wp.float32, "squeeze": wp.float32, "gain": wp.int64,
                "loss": wp.int64, "crossed": wp.int64, "flags": wp.int32, "ways_n": wp.int32, "adj_at": wp.int32,
                "adj_n": wp.int32, "way_at": wp.int32, "side_n": wp.int32}


class State:
    """The cells, their relations and their blocks of compute boxes."""

    def __init__(self, side: int, capacity: int, seed: int = 1, device: str = "cuda") -> None:
        if side < 4 or side % 2:
            raise ValueError(f"side must be even and at least 4, got {side}")
        self.side, self.lattice, self.device = int(side), int(side) ** 3, device
        if capacity < self.lattice:
            raise ValueError("capacity must hold every lattice place")
        self.capacity, self.count, self.step, self.seed = int(capacity), self.lattice, 0, int(seed)
        self.key = int(np.random.default_rng((int(seed), 11)).integers(0, 2 ** 31 - 1))
        self.quanta = wp.zeros((capacity, BLOCK, BLOCK, BLOCK), dtype=wp.int32, device=device)
        self.phase = wp.zeros((capacity, BLOCK, BLOCK, BLOCK), dtype=wp.float32, device=device)
        for name, dtype in _CELL_ARRAYS.items():
            setattr(self, name, wp.zeros(capacity, dtype=dtype, device=device))
        self.books = wp.zeros(4, dtype=wp.int64, device=device)
        self.found = wp.zeros(2, dtype=wp.int32, device=device)
        self.adj_pool = wp.zeros(1, dtype=wp.int32, device=device)
        self.way_pool = wp.zeros(1, dtype=wp.int32, device=device)
        self.kind_h = np.zeros(capacity, dtype=np.int8)
        self.ways_h = np.zeros(capacity, dtype=np.int32)
        places = np.arange(self.lattice)
        odd = (places // (side * side)) % 2 + (places // side) % 2 + places % 2
        self.kind_h[:self.lattice] = np.where(odd == 0, ENERGY_VOLUME, JOINT)
        self.ways_h[:self.lattice] = np.where(odd == 0, 6, np.where(odd == 1, 1, 0))
        self.adj_over, self.sides_over, self.fell_into = {}, {}, {}
        self.version = 0
        self.size_energy = {ENERGY_VOLUME: 0.0, JOINT: 0.0}
        self.start_quanta = {ENERGY_VOLUME: 0.0, JOINT: 0.0}
        self.start_length = {ENERGY_VOLUME: 0.0, JOINT: 0.0}
        self.collapses = self.merges = self.tears = self.torn = 0
        self.taken_joint = self.taken_volumes = 0
        self.start, self.quantum = None, None
        self.relink()

    def neighbours(self, cell: int) -> tuple:
        """A cell's neighbours and the direction code each lies in."""
        if cell in self.adj_over:
            return self.adj_over[cell]
        if not 0 <= cell < self.lattice:
            raise ValueError(f"cell {cell} has no relations")
        side = self.side
        place = np.array([cell // (side * side), (cell // side) % side, cell % side])
        codes = np.array([code for code in range(27) if code != CENTRE], dtype=np.int8)
        steps = np.stack([codes // 9 - 1, (codes // 3) % 3 - 1, codes % 3 - 1], axis=1)
        there = (place[None, :] + steps) % side
        return ((there[:, 0] * side + there[:, 1]) * side + there[:, 2]).astype(np.int32), codes

    def set_neighbours(self, cell: int, ids, codes) -> None:
        """Give a cell its own list of neighbours (no pair of id and code twice)."""
        keys = np.unique(np.asarray(ids, dtype=np.int64) * 32 + np.asarray(codes, dtype=np.int64))
        self.adj_over[int(cell)] = ((keys // 32).astype(np.int32), (keys % 32).astype(np.int8))

    def sides(self, joint: int) -> tuple:
        """The energy volumes on a joint's one side and on its other; every pair across is a way."""
        if joint in self.sides_over:
            return self.sides_over[joint]
        if not 0 <= joint < self.lattice or self.kind_h[joint] != JOINT:
            return [], []
        side = self.side
        place = [joint // (side * side), (joint // side) % side, joint % side]
        odd = [axis for axis in range(3) if place[axis] % 2]
        if len(odd) != 1:
            return [], []
        ends = []
        for shift in (-1, 1):
            there = list(place)
            there[odd[0]] = (there[odd[0]] + shift) % side
            ends.append((there[0] * side + there[1]) * side + there[2])
        return [ends[0]], [ends[1]]

    def ways(self, joint: int) -> list:
        """The pairs of energy volumes a joint lies between (energy crosses and presses along them)."""
        first, second = self.sides(joint)
        return [(near, far) for near in first for far in second if near != far]

    def set_sides(self, joint: int, first, second) -> None:
        """Give a joint its two sides (each volume once per side); the volumes' counts of ways follow."""
        self._count_ways(joint, -1)
        self.sides_over[int(joint)] = (list(dict.fromkeys(int(one) for one in first)),
                                       list(dict.fromkeys(int(one) for one in second)))
        self.ways_h[joint] = len(self.sides_over[joint][0]) * len(self.sides_over[joint][1])
        self._count_ways(joint, 1)

    def _count_ways(self, joint: int, sign: int) -> None:
        """Add (or take off) the joint's ways in the count of every volume on its sides."""
        first, second = self.sides(joint)
        both = set(first) & set(second)
        for mine, others in ((first, second), (second, first)):
            for volume in mine:
                self.ways_h[volume] += sign * (len(others) - (volume in both))

    def joints_of(self, cell: int) -> list:
        """The joints beside a cell, each once, in id order."""
        ids, _ = self.neighbours(cell)
        return sorted({int(one) for one in ids if self.kind_h[one] == JOINT})

    def block(self, cell: int) -> tuple:
        """Copies of one cell's boxes: (quanta int32 (6, 6, 6), phase float32 (6, 6, 6))."""
        return self.quanta[int(cell)].numpy(), self.phase[int(cell)].numpy()

    def set_block(self, cell: int, quanta: np.ndarray, phase: np.ndarray | None = None) -> None:
        """Write one cell's boxes; the caller keeps cell_quanta in step."""
        self.quanta[int(cell)].assign(np.ascontiguousarray(quanta, dtype=np.int32))
        if phase is not None:
            self.phase[int(cell)].assign(np.ascontiguousarray(phase, dtype=np.float32))

    def fill(self, cell: int, amount: int) -> None:
        """Stage: give a living cell exactly `amount` quanta, shared evenly over its boxes, phases kept."""
        if self.kind_h[cell] == 0 or amount < BOXES or amount >= BOXES * 2 ** 29:
            raise ValueError(f"cell {cell} cannot hold {amount} quanta")
        share = np.full(BOXES, amount // BOXES, dtype=np.int32)
        share[:amount % BOXES] += 1
        self.set_block(cell, int(self.kind_h[cell]) * share.reshape(BLOCK, BLOCK, BLOCK))
        held = self.cell_quanta.numpy()
        held[cell] = amount
        self.cell_quanta.assign(held)

    def host(self, name: str) -> np.ndarray:
        """A host copy of one per-cell array, ids 0 .. count - 1."""
        return getattr(self, name).numpy()[:self.count]

    def grow(self, needed: int) -> None:
        """Make room for at least `needed` cell ids; every id keeps its row and its boxes."""
        if needed <= self.capacity:
            return
        bigger = max(needed, self.capacity + self.capacity // 2)
        for name in ("quanta", "phase") + tuple(_CELL_ARRAYS):
            old = getattr(self, name)
            new = wp.zeros((bigger,) + tuple(old.shape[1:]), dtype=old.dtype, device=self.device)
            wp.copy(new, old, count=old.size)
            setattr(self, name, new)
        for name in ("kind_h", "ways_h"):
            old = getattr(self, name)
            new = np.zeros(bigger, dtype=old.dtype)
            new[:self.capacity] = old
            setattr(self, name, new)
        self.capacity = bigger

    def relink(self) -> None:
        """Copy the host's relations (kinds, own neighbour lists, ways) to the GPU."""
        adj_at = np.full(self.capacity, -1, dtype=np.int32)
        adj_n = np.full(self.capacity, 26, dtype=np.int32)
        way_at = np.full(self.capacity, -1, dtype=np.int32)
        pool, at = [], 0
        for cell, (ids, _) in self.adj_over.items():
            adj_at[cell], adj_n[cell] = at, len(ids)
            pool.append(ids)
            at += len(ids)
        side_n = np.ones(self.capacity, dtype=np.int32)
        ways, at = [], 0
        for joint, (first, second) in self.sides_over.items():
            way_at[joint], side_n[joint] = at, max(len(first), 1)
            ways.extend(first)
            ways.extend(second)
            at += len(first) + len(second)
        self.kind.assign(self.kind_h.astype(np.int32))
        self.ways_n.assign(self.ways_h)
        self.adj_at.assign(adj_at)
        self.adj_n.assign(adj_n)
        self.way_at.assign(way_at)
        self.side_n.assign(side_n)
        self.adj_pool = wp.array(np.concatenate(pool) if pool else np.zeros(1, dtype=np.int32), dtype=wp.int32,
                                 device=self.device)
        self.way_pool = wp.array(np.array(ways if ways else [0], dtype=np.int32), dtype=wp.int32, device=self.device)
