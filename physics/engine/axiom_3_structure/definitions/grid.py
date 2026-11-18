"""Grid: the one state object, the cells, their relations and each cell's own block of boxes.

Contract: `Grid(side, capacity, way_capacity, arrays, host, seed, device)` allocates an
empty state: one Warp array per name of `arrays` (`{name: (dtype, shape, each_step)}`,
every axiom's `ARRAYS` merged; shape "box" is (capacity, 6, 6, 6), "cell" (capacity,),
"way" (way_capacity,), a whole number n (n,)) and one host value per name of `host`
(deep-copied; `(dtype, "cell")` or `(dtype, "cell", fill)` is a per-cell numpy array that
grows with the grid). `arrays` must hold the structure's own names (structure_arrays.py).
The cells are the grid (axiom 3). A cell has no place: it has a kind, a length and
relations. Row `cell` of a "box" array is that cell's block for good; no box ever changes
cell and a used id is never used again.

Fixed attributes: `side` (cells along one edge of the start lattice, even, at least 4),
`capacity`, `way_capacity`, `count` (ids used so far; side ** 3 from the start),
`way_count` (way slots used so far), `seed`, `step` (0), `device`, `arrays`, `adj_id` and
`adj_code` (the neighbour pools on the device, owned and sized here).

Relations are tables.
  A neighbour entry of a cell is `(id, code)`: the neighbour and the direction it lies
  in, seen from the cell (direction_codes.py). The truth is the joint's list: when a
  joint lists volume v at `code`, v lists the joint at `opposite(code)`. Whoever changes
  one side sets the other too; nothing here mirrors for them.
  A way is a row `(a, b, mouth_a, mouth_b, joint)`: two entries `(a, code)` and
  `(b, opposite(code))` of one joint, both energy volumes, a != b, a the lower id;
  `mouth_a = opposite(code)` is the joint as seen from a and `mouth_b = code`. A way
  keeps its slot while it lives; a dead slot has `way_joint = -1` and may be given to a
  new way, which starts with 0 in every other "way" array.

Methods, all on the host:
  neighbours(cell) -> [(id, code)]        set_neighbours(cell, entries)
  ways_of(joint) -> [(slot, a, b, mouth_a, mouth_b)]
  block(name, cell) -> numpy (6, 6, 6)    set_block(name, cell, values)
  host(name) -> numpy copy of a whole "cell" or "way" array, the rows in use
  set_cell(name, cell, value)             one value of a "cell" array (device or host)
  new_ids(number) -> [id]                 grow(cells, ways): room for that many more
  free_slot() -> the lowest way slot holding no way (growing the table when full)
  send_neighbours(cells), send_rows(name, values, rows), send_pools()   upload host rows
What happens to relations is logic, not here: `start_grid` lays the start lattice and
`relink` (both in logic/) rebuilds the ways after `set_neighbours`. They work on the
host tables this class holds: `way_table`, `way_first`, `way_number`, `joint_slots`,
`free_slots`, `unsent`, and the neighbour pools `pool_id`, `pool_code`, `adj_start`,
`adj_count`, `adj_room`, `adj_used`.
"""
from __future__ import annotations

import copy
import heapq

import numpy as np
import warp as wp

from physics.engine.axiom_3_structure.definitions.direction_codes import CENTRE
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, ENERGY_VOLUME, JOINT
from physics.engine.axiom_3_structure.definitions.structure_arrays import ARRAYS as STRUCTURE_ARRAYS

_DTYPES = {"int32": wp.int32, "int64": wp.int64, "float32": wp.float32, "float64": wp.float64}
WAY_COLUMNS = ("way_a", "way_b", "way_mouth_a", "way_mouth_b", "way_joint", "way_pass")
JOINT_COLUMN = 4
PASS_COLUMN = 5
START_ENTRIES = 26
_FEW_ROWS = 64


def _is_cell_array(default) -> bool:
    """Whether a HOST default is the spec of a per-cell host array."""
    return isinstance(default, tuple) and len(default) in (2, 3) and default[0] in _DTYPES and default[1] == "cell"


class Grid:
    """The cells, their relations and their blocks of compute boxes."""

    def __init__(self, side: int, capacity: int, way_capacity: int, arrays: dict, host: dict, seed: int,
                 device: str) -> None:
        if side < 4 or side % 2:
            raise ValueError(f"side must be even and at least 4, got {side}")
        if capacity < side ** 3:
            raise ValueError(f"capacity {capacity} cannot hold the {side ** 3} cells of the start")
        missing = [name for name in STRUCTURE_ARRAYS if name not in arrays]
        if missing:
            raise ValueError(f"arrays lacks the structure's own {missing}")
        both = sorted(set(arrays) & set(host))
        if both:
            raise ValueError(f"names in both arrays and host: {both}")
        self.side, self.capacity, self.way_capacity = int(side), int(capacity), max(int(way_capacity), 1)
        self.count, self.way_count, self.seed, self.step, self.device = self.side ** 3, 0, int(seed), 0, device
        self.arrays = dict(arrays)
        self._host_specs = {name: default for name, default in host.items() if _is_cell_array(default)}
        for name, (dtype, shape, each_step) in self.arrays.items():
            setattr(self, name, wp.zeros(self._shape(shape), dtype=_DTYPES[dtype], device=device))
        for name, default in host.items():
            setattr(self, name, self._host_array(default, self.capacity) if name in self._host_specs
                    else copy.deepcopy(default))
        self.way_table = np.full((self.way_capacity, len(WAY_COLUMNS)), -1, dtype=np.int32)
        self.way_joint.assign(self.way_table[:, JOINT_COLUMN])
        self.way_first = np.zeros(self.capacity, dtype=np.int64)
        self.way_number = np.zeros(self.capacity, dtype=np.int32)
        self.joint_slots: dict = {}
        self.free_slots: list = []
        self.adj_start = np.zeros(self.capacity, dtype=np.int32)
        self.adj_count = np.zeros(self.capacity, dtype=np.int32)
        self.adj_room = np.zeros(self.capacity, dtype=np.int32)
        self.adj_used = 0
        self.pool_id = np.zeros(max(self.capacity * START_ENTRIES, 1), dtype=np.int32)
        self.pool_code = np.zeros_like(self.pool_id)
        self.adj_id = wp.zeros(self.pool_id.size, dtype=wp.int32, device=device)
        self.adj_code = wp.zeros(self.pool_id.size, dtype=wp.int32, device=device)
        self.unsent: set = set()

    def _shape(self, shape) -> tuple:
        """The array shape a shape word of ARRAYS stands for."""
        if shape == "box":
            return (self.capacity, BLOCK, BLOCK, BLOCK)
        if shape == "cell":
            return (self.capacity,)
        if shape == "way":
            return (self.way_capacity,)
        if isinstance(shape, int) and not isinstance(shape, bool) and shape > 0:
            return (shape,)
        raise ValueError(f"unknown array shape {shape!r}")

    @staticmethod
    def _host_array(spec: tuple, room: int) -> np.ndarray:
        """A per-cell host array of `room` rows holding its spec's empty value."""
        return np.full(room, spec[2] if len(spec) == 3 else 0, dtype=spec[0])

    def _check_id(self, cell: int) -> int:
        """The id as an int; raises for one that was never handed out."""
        cell = int(cell)
        if not 0 <= cell < self.count:
            raise ValueError(f"cell id {cell} was never handed out (ids used: {self.count})")
        return cell

    def neighbours(self, cell: int) -> list:
        """A cell's neighbour entries, `(id, code)` each."""
        cell = self._check_id(cell)
        first = int(self.adj_start[cell])
        last = first + int(self.adj_count[cell])
        return list(zip(self.pool_id[first:last].tolist(), self.pool_code[first:last].tolist()))

    def set_neighbours(self, cell: int, entries) -> None:
        """Give a cell its neighbour entries on the host (no entry twice); `relink` uploads them."""
        cell = self._check_id(cell)
        entries = list(dict.fromkeys((int(one), int(code)) for one, code in entries))
        for one, code in entries:
            if not 0 <= one < self.count or not 0 <= code <= 26 or code == CENTRE or one == cell:
                raise ValueError(f"cell {cell} cannot have the neighbour entry {(one, code)}")
        if len(entries) > self.adj_room[cell]:
            self.room_in_pool(len(entries))
            self.adj_start[cell], self.adj_room[cell] = self.adj_used, len(entries)
            self.adj_used += len(entries)
        first = int(self.adj_start[cell])
        self.adj_count[cell] = len(entries)
        self.pool_id[first:first + len(entries)] = [entry[0] for entry in entries]
        self.pool_code[first:first + len(entries)] = [entry[1] for entry in entries]
        self.unsent.add(cell)

    def room_in_pool(self, more: int) -> None:
        """Make the host pools hold `more` further entries."""
        needed = self.adj_used + more
        if needed <= self.pool_id.size:
            return
        bigger = max(needed, self.pool_id.size + self.pool_id.size // 2)
        for name in ("pool_id", "pool_code"):
            grown = np.zeros(bigger, dtype=np.int32)
            grown[:self.adj_used] = getattr(self, name)[:self.adj_used]
            setattr(self, name, grown)

    def slots_of(self, joint: int) -> list:
        """The way slots a joint holds now."""
        if joint in self.joint_slots:
            return self.joint_slots[joint]
        first = int(self.way_first[joint])
        return list(range(first, first + int(self.way_number[joint])))

    def ways_of(self, joint: int) -> list:
        """A joint's living ways, `(slot, a, b, mouth_a, mouth_b)` each, in slot order."""
        joint = self._check_id(joint)
        return [(slot,) + tuple(int(value) for value in self.way_table[slot, :JOINT_COLUMN])
                for slot in sorted(self.slots_of(joint))]

    def block(self, name: str, cell: int) -> np.ndarray:
        """A copy of one cell's block of a "box" array."""
        if self.arrays[name][1] != "box":
            raise ValueError(f"{name} is not a box array")
        return getattr(self, name)[self._check_id(cell)].numpy().copy()

    def set_block(self, name: str, cell: int, values) -> None:
        """Write one cell's block of a "box" array, nothing more."""
        dtype, shape = self.arrays[name][:2]
        if shape != "box":
            raise ValueError(f"{name} is not a box array")
        values = np.ascontiguousarray(values, dtype=dtype)
        if values.shape != (BLOCK, BLOCK, BLOCK):
            raise ValueError(f"a block is {(BLOCK, BLOCK, BLOCK)}, got {values.shape}")
        getattr(self, name)[self._check_id(cell)].assign(values)

    def host(self, name: str) -> np.ndarray:
        """A host copy of a whole "cell" or "way" array: the rows in use (a short array whole)."""
        if name in self._host_specs:
            return getattr(self, name)[:self.count].copy()
        shape = self.arrays[name][1]
        if shape == "box":
            raise ValueError(f"{name} is a box array; read one cell with block()")
        values = getattr(self, name).numpy()
        if shape == "cell":
            values = values[:self.count]
        elif shape == "way":
            values = values[:self.way_count]
        return values.copy()

    def set_cell(self, name: str, cell: int, value) -> None:
        """Write one cell's value of a "cell" array, on the device or on the host."""
        cell = self._check_id(cell)
        if name in self._host_specs:
            getattr(self, name)[cell] = value
            return
        dtype, shape = self.arrays[name][:2]
        if shape != "cell":
            raise ValueError(f"{name} is not a cell array")
        getattr(self, name)[cell:cell + 1].assign(np.array([value], dtype=dtype))

    def new_ids(self, number: int) -> list:
        """Hand out `number` ids never used before."""
        self.grow(number, 0)
        ids = list(range(self.count, self.count + int(number)))
        self.count += int(number)
        return ids

    def grow(self, cells: int, ways: int) -> None:
        """Make room for `cells` more ids and `ways` more way slots; every row in use is kept."""
        if self.count + int(cells) > self.capacity:
            self._grow_cells(max(self.count + int(cells), self.capacity + self.capacity // 2))
        if self.way_count + int(ways) > self.way_capacity:
            self._grow_ways(max(self.way_count + int(ways), self.way_capacity + self.way_capacity // 2))

    def _grow_cells(self, capacity: int) -> None:
        """Move every per-cell and per-box array to `capacity` rows."""
        self.capacity = capacity
        for name, (dtype, shape, each_step) in self.arrays.items():
            if shape in ("box", "cell"):
                old = getattr(self, name)
                new = wp.zeros(self._shape(shape), dtype=_DTYPES[dtype], device=self.device)
                wp.copy(new, old, count=old.size)
                setattr(self, name, new)
        for name, spec in self._host_specs.items():
            setattr(self, name, self._longer(getattr(self, name), self._host_array(spec, capacity)))
        for name in ("way_first", "way_number", "adj_start", "adj_count", "adj_room"):
            old = getattr(self, name)
            setattr(self, name, self._longer(old, np.zeros(capacity, dtype=old.dtype)))

    @staticmethod
    def _longer(old: np.ndarray, new: np.ndarray) -> np.ndarray:
        """`new` with the rows of `old` copied into its start."""
        new[:len(old)] = old
        return new

    def _grow_ways(self, way_capacity: int) -> None:
        """Move every per-way array to `way_capacity` slots; the new slots hold no way."""
        self.way_capacity = way_capacity
        table = np.full((way_capacity, len(WAY_COLUMNS)), -1, dtype=np.int32)
        self.way_table = self._longer(self.way_table, table)
        for name, (dtype, shape, each_step) in self.arrays.items():
            if shape == "way":
                old = getattr(self, name)
                new = wp.zeros(way_capacity, dtype=_DTYPES[dtype], device=self.device)
                wp.copy(new, old, count=old.size)
                setattr(self, name, new)
        self.way_joint.assign(self.way_table[:, JOINT_COLUMN])

    def send_pools(self) -> None:
        """Upload the whole neighbour pools, making the device's as long as the host's."""
        if self.adj_id.size != self.pool_id.size:
            self.adj_id = wp.zeros(self.pool_id.size, dtype=wp.int32, device=self.device)
            self.adj_code = wp.zeros(self.pool_id.size, dtype=wp.int32, device=self.device)
        self.adj_id.assign(self.pool_id)
        self.adj_code.assign(self.pool_code)

    def send_rows(self, name: str, values: np.ndarray, rows: list) -> None:
        """Upload the named rows of one array (the whole array when they are many)."""
        target = getattr(self, name)
        if len(rows) > _FEW_ROWS:
            target.assign(np.ascontiguousarray(values))
            return
        for row in rows:
            target[row:row + 1].assign(np.ascontiguousarray(values[row:row + 1]))

    def send_neighbours(self, cells: list) -> None:
        """Upload the neighbour rows of the named cells."""
        if self.adj_id.size != self.pool_id.size or len(cells) > _FEW_ROWS:
            self.send_pools()
        else:
            for cell in cells:
                first, last = int(self.adj_start[cell]), int(self.adj_start[cell]) + int(self.adj_count[cell])
                if last > first:
                    self.adj_id[first:last].assign(self.pool_id[first:last])
                    self.adj_code[first:last].assign(self.pool_code[first:last])
        self.send_rows("adj_at", self.adj_start, cells)
        self.send_rows("adj_n", self.adj_count, cells)

    def free_slot(self) -> int:
        """The lowest way slot holding no way, growing the table when it is full."""
        if self.free_slots:
            return heapq.heappop(self.free_slots)
        self.grow(0, 1)
        self.way_count += 1
        return self.way_count - 1
