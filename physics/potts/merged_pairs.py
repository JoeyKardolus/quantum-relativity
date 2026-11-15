"""Merged pairs: add touching-pair counts together, as joint collapse and volume split change them.

Contract: `merged_pairs(parts)` takes a list of (first, second, faces) int numpy triples
(cell ids in any order, never equal; faces may be negative, to take faces away) and
returns one (low, high, faces) int64 triple, low < high, sorted, each pair once with its
faces summed; pairs whose faces come to 0 or less are dropped (they no longer touch).
"""
from __future__ import annotations

import numpy as np


def merged_pairs(parts: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One sorted (low, high, faces) triple from many, faces summed per pair."""
    first = np.concatenate([np.asarray(part[0], dtype=np.int64) for part in parts])
    second = np.concatenate([np.asarray(part[1], dtype=np.int64) for part in parts])
    faces = np.concatenate([np.asarray(part[2], dtype=np.int64) for part in parts])
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    if low.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    span = int(high.max()) + 1
    keys, inverse = np.unique(low * span + high, return_inverse=True)
    totals = np.bincount(inverse, weights=faces, minlength=keys.size).astype(np.int64)
    kept = totals > 0
    return keys[kept] // span, keys[kept] % span, totals[kept]
