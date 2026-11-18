"""Memory law: how much GPU memory a universe of a size needs, so a probe is refused before it is built.

The engine holds `cells_per_side ** 3 * (1 + spare)` cell ids (the spare ones wait for
tears), each with its 216 compute boxes. `BYTES_PER_CELL_ID` is the most Warp's pool held
per id after six steps of the default universe, measured 2026-11-18 on the rebuilt
engine: 3,046 at 8 and at 16 cells a side, 3,150 at 32. The largest is kept, so the
guard errs towards refusing. It is a measurement, not a setting.
  `bytes_needed(cells_per_side, spare=SPARE)`      ids times BYTES_PER_CELL_ID
  `fits(cells_per_side, free_bytes, spare=SPARE)`  True when bytes_needed is at most free_bytes
"""
from __future__ import annotations

from physics.bench.throughput.definitions.engine_start import SPARE

BYTES_PER_CELL_ID = 3150


def bytes_needed(cells_per_side: int, spare: float = SPARE) -> float:
    """GPU bytes a universe of `cells_per_side` cells a side holds, spare ids included."""
    return cells_per_side ** 3 * (1.0 + spare) * BYTES_PER_CELL_ID


def fits(cells_per_side: int, free_bytes: float, spare: float = SPARE) -> bool:
    """True when the universe's bytes are no more than the free bytes."""
    return bytes_needed(cells_per_side, spare) <= free_bytes
