"""Take snapshot: the cells of one moment, kept so the lumps found at the end can be followed back.

Contract: `take_snapshot(cells)` takes a `read_universe` dict and returns
  held     every cell id's quanta (int64)
  length   every cell id's length (float64); a cell that is gone or holds nothing counts
           0, as it is left out of the universe's size
  count    the ids in use
  plus     the quanta in all energy volumes
  lumps    the ids of the lump volumes
  ways     `(joint, first, second)`, shared with the read
  version  the grid's count of tears and collapses
A cell id is the same cell for good, so a volume found at the end is the same row in
every earlier snapshot.
"""
from __future__ import annotations

import numpy as np

from physics.engine import ENERGY_VOLUME
from physics.fit.measures.definitions.lump_volumes import lump_volumes


def take_snapshot(cells: dict) -> dict:
    """The cells' quanta and lengths, the lump volumes and the ways of one read."""
    kind, held = cells["kind"], cells["held"]
    return {"held": held, "length": np.where(np.isfinite(cells["length"]), cells["length"], 0.0),
            "count": int(cells["count"]), "plus": int(held[kind == ENERGY_VOLUME].sum()),
            "lumps": np.flatnonzero(lump_volumes(kind, held, cells["start_volume_quanta"])),
            "ways": cells["ways"], "version": int(cells["version"])}
