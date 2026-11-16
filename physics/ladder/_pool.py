"""Slot pools: fixed-size rows of device arrays, handed out and taken back by slot number.

`_Pool(specs, device)` holds one device array per name, shape (capacity,) + row shape;
`alloc(count)` returns that many free slot numbers (growing every array to a quarter
more than needed, old rows copied), `release(slots)` takes them back. A slot's row is
undefined until written. `shrink(live)` gives the memory of released rows back: when
fewer than half the rows are in use it moves the live rows down to slots 0.. in order,
reallocates every array a quarter above the live count (at least 16 rows) and returns
the old-to-new slot map (-1 for a slot not live), so the caller renumbers its table;
otherwise it returns None and nothing moves. Rows are moved whole, bit for bit.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

_FLOOR = 16


@wp.kernel
def _take_rows(source: wp.array2d(dtype=Any), rows: wp.array(dtype=wp.int32), target: wp.array2d(dtype=Any)):
    """target[n] = source[rows[n]], element by element."""
    item, place = wp.tid()
    target[item, place] = source[rows[item], place]


class _Pool:
    """Device rows handed out by slot number."""

    def __init__(self, specs: dict, device, capacity: int = _FLOOR):
        self.specs = specs
        self.device = device
        self.capacity = 0
        self.arrays: dict = {}
        self.free: list[int] = []
        self._grow(capacity)

    def _grow(self, capacity: int) -> None:
        """Reallocate every array at `capacity` rows, keeping the old rows."""
        for name, (dtype, row) in self.specs.items():
            fresh = wp.zeros((capacity,) + tuple(row), dtype=dtype, device=self.device)
            if name in self.arrays and self.capacity:
                wp.copy(fresh, self.arrays[name], count=self.arrays[name].size)
            self.arrays[name] = fresh
        self.free.extend(range(capacity - 1, self.capacity - 1, -1))
        self.capacity = capacity

    def alloc(self, count: int) -> np.ndarray:
        """`count` free slot numbers, growing the pool when it runs out."""
        if count > len(self.free):
            needed = self.capacity - len(self.free) + count
            self._grow(max(_FLOOR, needed + needed // 4))
        taken = self.free[len(self.free) - count:]
        del self.free[len(self.free) - count:]
        return np.asarray(taken[::-1], dtype=np.int32)

    def release(self, slots) -> None:
        """Take slots back."""
        self.free.extend(int(slot) for slot in np.asarray(slots).ravel())

    def shrink(self, live) -> np.ndarray | None:
        """Compact the live slots to 0.. and free the rest when under half the rows are in use; the slot map or None."""
        live = np.sort(np.asarray(live, dtype=np.int32).ravel())
        if live.size != self.in_use():
            raise ValueError(f"{live.size} live slots named, {self.in_use()} in use")
        capacity = max(_FLOOR, live.size + live.size // 4)
        if 2 * live.size >= self.capacity or capacity >= self.capacity:
            return None
        rows = wp.array(live, dtype=wp.int32, device=self.device)
        for name, (dtype, row) in self.specs.items():
            width = int(np.prod(row))
            fresh = wp.zeros((capacity,) + tuple(row), dtype=dtype, device=self.device)
            if live.size:
                wp.launch(_take_rows, dim=(live.size, width), device=self.device,
                          inputs=[self.arrays[name].reshape((self.capacity, width)), rows],
                          outputs=[fresh.reshape((capacity, width))])
            self.arrays[name] = fresh
        remap = np.full(self.capacity, -1, dtype=np.int32)
        remap[live] = np.arange(live.size, dtype=np.int32)
        self.capacity = capacity
        self.free = list(range(capacity - 1, live.size - 1, -1))
        return remap

    def in_use(self) -> int:
        """Slots handed out and not taken back."""
        return self.capacity - len(self.free)

    def slot_bytes(self) -> int:
        """Bytes one slot's rows take over all arrays."""
        return sum(int(np.prod(row)) * wp.types.type_size_in_bytes(dtype) for dtype, row in self.specs.values())
