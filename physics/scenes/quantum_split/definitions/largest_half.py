"""Largest half: for each open split, the cell holding most of it and the share it holds.

Contract: `largest_half(sources, part_source, part_cell, part_quanta)` takes the open
splits' source volumes (rising) and the store's halves (three equal-length int columns:
the split, the cell, the whole quanta there; physics.engine split store). Returns
`(cells, shares)`, two arrays as long as `sources`: the cell of each split's largest
half (a tie to the lower cell id) and that half's share of the split's quanta, in
(0, 1]. A split with no quanta gets cell -1 and share 0. Raises ValueError for columns
of unequal length or a half whose split is not in `sources`.
"""
from __future__ import annotations

import numpy as np


def largest_half(sources, part_source, part_cell, part_quanta) -> tuple[np.ndarray, np.ndarray]:
    """Per open split, its largest half's cell and that half's share of the split."""
    sources = np.asarray(sources, dtype=np.int64)
    part_source, part_cell = np.asarray(part_source, dtype=np.int64), np.asarray(part_cell, dtype=np.int64)
    part_quanta = np.asarray(part_quanta, dtype=np.int64)
    if not part_source.shape == part_cell.shape == part_quanta.shape:
        raise ValueError("the halves' three columns must be equally long")
    row = np.searchsorted(sources, part_source)
    if part_source.size and (row.max(initial=0) >= sources.size or np.any(sources[row] != part_source)):
        raise ValueError("a half names a split that is not open")
    cells, largest, total = np.full(sources.size, -1, dtype=np.int64), np.zeros(sources.size), np.zeros(sources.size)
    for split, cell, quanta in sorted(zip(row.tolist(), part_cell.tolist(), part_quanta.tolist())):
        total[split] += quanta
        if quanta > largest[split]:
            cells[split], largest[split] = cell, quanta
    return cells, np.divide(largest, total, out=np.zeros(sources.size), where=total > 0)
