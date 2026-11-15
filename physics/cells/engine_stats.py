"""Engine stats: what the grid looks like now.

Contract: `engine_stats(engine)` returns a dict:
  cells          energy volumes now
  joints         joints now (pairs of neighbouring energy volumes, one joint cell each)
  joint_cells    cells of joint material now: joint cells, seam cells and corner
                 cells (cell_complex.py)
  corners        corners of all cells now
  collapses      joint collapses since the start
  splits         volume splits since the start
  size_vs_start  the volume of all cells now / at the start (cell_sizes.py summed: all
                 of space, energy volumes and joint material together)
  total_energy   sum of all energy (never changes)
  lumps          lumps now (lumps.py: packed groups of neighbouring energy volumes)
  sealed_lumps   lumps whose outer seams are all closed (lumps.py)
  neighbour_changes  neighbour pairs changed outside a collapse or split (0 by construction)
"""
from __future__ import annotations

from physics.cells.cell_sizes import cell_sizes
from physics.cells.engine import Engine
from physics.cells.lumps import lumps


def engine_stats(engine: Engine) -> dict:
    """Cells, joints, joint cells, corners, collapses, splits, size against the start, energy, lumps, sealed, neighbour changes."""
    grid = engine.grid
    _, sealed = lumps(grid)
    space = float(cell_sizes(engine.corners, engine.faces, engine.face_cells, len(engine.cell_kind)).sum())
    return {"cells": grid.cell_count,
            "joints": len(grid.joints),
            "joint_cells": len(engine.cell_kind) - grid.cell_count,
            "corners": len(engine.corners),
            "collapses": engine.collapses,
            "splits": engine.splits,
            "size_vs_start": space / engine.start_space,
            "total_energy": grid.total_energy(),
            "lumps": len(sealed),
            "sealed_lumps": int(sealed.sum()),
            "neighbour_changes": engine.neighbour_changes}
