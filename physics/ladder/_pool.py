"""Slot pools: fixed-size rows of device arrays, handed out and taken back by slot number.

`_Pool(specs, device)` holds one device array per name, shape (capacity,) + row shape;
`alloc(count)` returns that many free slot numbers (growing every array by doubling, old
rows copied), `release(slots)` takes them back. A slot's row is undefined until written.
"""
from __future__ import annotations

import numpy as np
import warp as wp


class _Pool:
    """Device rows handed out by slot number."""

    def __init__(self, specs: dict, device, capacity: int = 16):
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
            capacity = max(self.capacity, 16)
            while capacity < needed:
                capacity *= 2
            self._grow(capacity)
        taken = self.free[len(self.free) - count:]
        del self.free[len(self.free) - count:]
        return np.asarray(taken[::-1], dtype=np.int32)

    def release(self, slots) -> None:
        """Take slots back."""
        self.free.extend(int(slot) for slot in np.asarray(slots).ravel())

    def in_use(self) -> int:
        """Slots handed out and not taken back."""
        return self.capacity - len(self.free)

    def slot_bytes(self) -> int:
        """Bytes one slot's rows take over all arrays."""
        return sum(int(np.prod(row)) * wp.types.type_size_in_bytes(dtype) for dtype, row in self.specs.values())
