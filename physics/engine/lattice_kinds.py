"""Lattice kinds: every cell's kind in the forced cell lattice, on the host (numpy mirror of cell_kind.py).

Contract: `lattice_kinds(cells)` takes the real cells per axis, one int for a cube or
three ints (nx, ny, nz), and returns an int8 array of shape (nx + 2, ny + 2, nz + 2)
indexed by the extended lattice indices (a, b, c) (cell_place.py): 1 energy volume
(all three indices even, in 2 .. n - 1 on each axis), -1 joint (every other real cell),
0 the ring of empty space round it. `.ravel()` gives it per cell id. Raises ValueError
for fewer than 3 cells along an axis.
"""
from __future__ import annotations

import numpy as np


def lattice_kinds(cells) -> np.ndarray:
    """Per extended-lattice cell: 1 energy volume, -1 joint, 0 ring."""
    counts = (int(cells),) * 3 if np.ndim(cells) == 0 else tuple(int(count) for count in cells)
    if len(counts) != 3 or min(counts) < 3:
        raise ValueError(f"a cell lattice needs at least 3 cells along each axis, got {cells}")
    real, volume = [], []
    for count in counts:
        index = np.arange(count + 2)
        real.append((index >= 1) & (index <= count))
        volume.append((index % 2 == 0) & (index >= 2) & (index <= count - 1))
    inside = real[0][:, None, None] & real[1][None, :, None] & real[2][None, None, :]
    volumes = volume[0][:, None, None] & volume[1][None, :, None] & volume[2][None, None, :]
    return np.where(volumes, 1, np.where(inside, -1, 0)).astype(np.int8)
