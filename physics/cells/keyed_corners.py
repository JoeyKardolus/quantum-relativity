"""Keyed corners: where each corner of freshly cut cells was before, found by its key.

Contract: `keyed_corners(keys, old_keys, old_corners)` returns (k, 3): per key (rows of
cell_complex.py keys) the old position of the corner with the same key in old_keys
(already renumbered into the new skeleton), NaN where there was none (a corner a
collapse or split made new). Exact while 4 * (skeleton corners + 2) ** 4 fits in int64
(up to about 40 000 skeleton corners); raises ValueError past that.
"""
from __future__ import annotations

import numpy as np


def keyed_corners(keys: np.ndarray, old_keys: np.ndarray, old_corners: np.ndarray) -> np.ndarray:
    """Old positions of the corners with the same keys, NaN for new ones."""
    width = int(max(keys.max(initial=0), old_keys.max(initial=0))) + 2
    if 4 * float(width) ** 4 >= 2.0 ** 63:
        raise ValueError("keyed_corners: too many skeleton corners for an exact key code")
    new_code, old_code = _code(keys, width), _code(old_keys, width)
    order = np.argsort(old_code)
    found = np.minimum(np.searchsorted(old_code[order], new_code), max(len(order) - 1, 0))
    positions = np.full((len(keys), 3), np.nan)
    if not len(order):
        return positions
    hit = old_code[order][found] == new_code
    positions[hit] = old_corners[order[found[hit]]]
    return positions


def _code(keys: np.ndarray, width: int) -> np.ndarray:
    """One int64 per key row: the kind (0..3), then each corner shifted by one so -1 fits."""
    code = np.zeros(len(keys), dtype=np.int64)
    for column in range(1, keys.shape[1]):
        code = code * width + (keys[:, column] + 1)
    return code * 4 + keys[:, 0]
