"""Read universe: the cells and ways of one moment, copied to the host for the fit's measures.

Contract: `read_universe(universe, before=None)` reads a `physics.engine` `Grid` and
returns a dict of host copies; it writes nothing.
  side, count    cells along the start lattice's edge; cell ids handed out so far
  kind           int32 per cell id: 1 energy volume, -1 joint, 0 no cell
  held           int64 per cell: its quanta, as a size
  length         float64 per cell: its length by the size law (inf for a cell holding 0)
  rate           float64 per cell: its clock rate
  ways           `(joint, first, second)` int64 arrays, one row per living way. When
                 `before` is the previous read and no collapse or tear happened since
                 (the same `version`), its ways are shared, not read again
  fell_into      int32 per cell: the volume a vanished volume fell into, -1 for none
  start_volume_quanta   the start's plus quanta over the start's energy volumes
  version, collapses, merges, tears   the grid's counts since the start
  squeeze_max    the hardest-squeezed joint's press over its push this step (0.0 with no joint)
  packing_max    the most packed energy volume's packing (0.0 with no volume)
A cell id is the same cell for good, so a row means the same cell in every read of a run.
"""
from __future__ import annotations

import numpy as np

from physics.engine import ENERGY_VOLUME, JOINT, Grid, read_ways


def _ways(universe: Grid) -> tuple:
    """The living ways as `(joint, first, second)` arrays."""
    rows = read_ways(universe).astype(np.int64)
    return rows[:, 3], rows[:, 1], rows[:, 2]


def _largest(values: np.ndarray) -> float:
    """The largest of some values, 0.0 for none."""
    return float(values.max()) if values.size else 0.0


def read_universe(universe: Grid, before: dict | None = None) -> dict:
    """Host copies of the cells, the ways and the counts the fit measures from."""
    kind = universe.host("kind")
    same = before is not None and before["version"] == universe.version
    return {"side": int(universe.side), "count": int(universe.count), "kind": kind,
            "held": universe.host("cell_quanta").astype(np.int64),
            "length": universe.host("length").astype(np.float64),
            "rate": universe.host("rate").astype(np.float64),
            "ways": before["ways"] if same else _ways(universe),
            "fell_into": universe.host("fell_into"),
            "start_volume_quanta": universe.start_books["plus"] / (universe.side // 2) ** 3,
            "version": int(universe.version), "collapses": int(universe.collapses),
            "merges": int(universe.merges), "tears": int(universe.tears),
            "squeeze_max": _largest(universe.host("squeeze")[kind == JOINT]),
            "packing_max": _largest(universe.host("packing")[kind == ENERGY_VOLUME])}
