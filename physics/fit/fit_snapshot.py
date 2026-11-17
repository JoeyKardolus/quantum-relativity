"""fit_snapshot: the raw cells of one moment, kept so that the lumps found at the end can be followed back.

Contract: fit_snapshot(universe, before=None) -> dict of host arrays for a
physics.engine State now:
  held    every cell id's quanta (int64), length its length (float32)
  lumps   the ids of the lump volumes (fit_measure's lump_volumes); plus the quanta in
          all energy volumes
  ways    (joint, first volume, second volume) int64 arrays, one row per way; when
          `before` is the previous snapshot and no collapse or tear happened since
          (the same state.version), its ways are shared, not read again
  version state.version, count the ids in use
A cell id is the same cell for good (physics.engine), so a volume found at the end is
the same row in every earlier snapshot.
"""
from __future__ import annotations

import numpy as np

from physics.engine import ENERGY_VOLUME, State, way_list
from physics.engine.stats import LUMP_FACTOR


def fit_snapshot(universe: State, before: dict | None = None) -> dict:
    """The cells' quanta and lengths, the lump volumes and the ways, as arrays."""
    held = universe.host("cell_quanta").astype(np.int64)
    volumes = universe.kind_h[:universe.count] == ENERGY_VOLUME
    same = before is not None and before["version"] == universe.version
    return {"held": held, "length": universe.host("length").copy(), "count": int(universe.count),
            "plus": int(held[volumes].sum()),
            "lumps": np.flatnonzero(volumes & (held >= LUMP_FACTOR * universe.start_quanta[ENERGY_VOLUME])),
            "ways": before["ways"] if same else way_list(universe), "version": int(universe.version)}
