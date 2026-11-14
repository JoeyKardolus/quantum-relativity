"""Engine stats: what the grid looks like now, read from energies, sizes and the neighbour graph only.

Contract: `engine_stats(engine)` returns a dict:
  cells          energy volumes now
  joints         joints now (pairs of neighbouring energy volumes)
  collapses      energy volumes that fell into others since the start
  size_vs_start  total_space() now / at the start (energy volumes plus joints)
  energy_spread  std / mean of energy per unit volume (packing)
  lumps          lumps now (lumps.py: packed groups of neighbouring energy volumes)
  sealed_lumps   lumps whose outer seams are all closed (lumps.py)
  neighbour_changes  neighbour pairs the drawn shapes changed by themselves, outside a
                 joint collapse, since the start (Engine.neighbour_changes); the grid
                 never follows them, 0 when the shapes stay woven too
No position, weight or shape is read.
"""
from __future__ import annotations

from physics.cells.engine import Engine
from physics.cells.lumps import lumps


def engine_stats(engine: Engine) -> dict:
    """Cells, joints, collapses, size against the start, packing spread, lumps, sealed, neighbour changes."""
    grid = engine.grid
    packing = grid.energy / grid.volume
    _, sealed = lumps(grid)
    return {"cells": grid.cell_count,
            "joints": len(grid.joints),
            "collapses": engine.collapses,
            "size_vs_start": float(grid.total_space() / engine.start_space),
            "energy_spread": float(packing.std() / packing.mean()),
            "lumps": len(sealed),
            "sealed_lumps": int(sealed.sum()),
            "neighbour_changes": engine.neighbour_changes}
