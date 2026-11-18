"""Landing: how often settlings landed in their split's largest half, beside the share those halves held.

Contract: `landing(settled_source, settled_cell, sources, cells, shares)` takes the
settlings (two equal-length int columns: the split's source volume and the cell it
settled whole in) and, per split open at the reading, its largest half's cell and share
(largest_half). Returns a dict:
  judged     settlings whose split was open at the reading
  unread     settlings whose split was not (no share on record; left out)
  found      the share of the judged settlings that landed in their largest half's cell
  expected   the mean share those largest halves held: what `found` must be when a
             cell's share is its chance
`found` and `expected` are 0.0 with nothing judged.
"""
from __future__ import annotations

import numpy as np


def landing(settled_source, settled_cell, sources, cells, shares) -> dict:
    """The share of settlings found in the largest half's cell, and the share expected there."""
    settled_source = np.asarray(settled_source, dtype=np.int64)
    settled_cell = np.asarray(settled_cell, dtype=np.int64)
    sources = np.asarray(sources, dtype=np.int64)
    row = np.minimum(np.searchsorted(sources, settled_source), max(sources.size - 1, 0))
    read = (sources[row] == settled_source) if sources.size else np.zeros(settled_source.size, dtype=bool)
    row = row[read]
    judged = int(read.sum())
    return {"judged": judged, "unread": int(settled_source.size - judged),
            "found": float((settled_cell[read] == np.asarray(cells)[row]).mean()) if judged else 0.0,
            "expected": float(np.asarray(shares)[row].mean()) if judged else 0.0}
