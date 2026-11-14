"""Neighbour changes: neighbour pairs the shapes changed by themselves, outside any joint collapse.

Contract: `neighbour_changes(before, merged_into, after, joints)` returns an int:
the pairs of cells that neighbour in one tessellation and not in the other,
  before       (k, 2) the old tessellation's pairs (Tessellation.pairs), old indices
  merged_into  (n_old,) each old energy volume's index now (Grid.merged_into)
  after        (m, 2), a < b: the new tessellation's pairs, new indices
  joints       (j, 2) the grid's joints now: the woven neighbour graph
`before` is relabelled through merged_into first (pairs inside a volume that fell
into one dropped), and a pair with an end on a volume that fell into another this
step, or on one of its neighbours in `joints`, is not counted: the collapse
reshaped those. The grid itself changes neighbours only by joint collapse
(apply_rules.py); this counts what the drawn shapes did beside it, 0 when they
stayed woven. Reads only index arrays, never a position.
"""
from __future__ import annotations

import numpy as np


def neighbour_changes(before: np.ndarray, merged_into: np.ndarray, after: np.ndarray,
                      joints: np.ndarray) -> int:
    """Pairs that appeared or vanished between two tessellations, collapse zones left out."""
    count = int(merged_into.max()) + 1 if len(merged_into) else 0
    relabelled = np.sort(merged_into[before], axis=1)
    relabelled = relabelled[relabelled[:, 0] != relabelled[:, 1]]
    keys = np.setxor1d(relabelled[:, 0].astype(np.int64) * count + relabelled[:, 1],
                       after[:, 0].astype(np.int64) * count + after[:, 1])
    ends = np.column_stack([keys // count, keys % count])
    loose = _collapse_zone(merged_into, joints, count)
    return int((~loose[ends].any(axis=1)).sum())


def _collapse_zone(merged_into: np.ndarray, joints: np.ndarray, count: int) -> np.ndarray:
    """Volumes made of several old ones this step, and their neighbours in `joints`."""
    merged = np.bincount(merged_into, minlength=count) > 1
    zone = merged.copy()
    zone[joints[merged[joints].any(axis=1)].ravel()] = True
    return zone
