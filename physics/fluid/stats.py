"""Stats: what the energy fluid looks like now (copies the energies to the host once).

Contract: `stats(grid, settings=None)` (default Settings.all_on()) returns a dict:
  total_energy      A- plus B-energy over all boxes (float)
  energy_drift      total now minus at the start, in quanta (int; 0 is exact)
  occupied          boxes holding energy above occupied_floor * start_a_mean
  size_vs_start     occupied now / occupied at the start (volume of the occupied region)
  lumps             6-connected regions of boxes with A-energy >= lump_factor * start_a_mean
  sealed_lumps      lumps whose every surface box (one with a face neighbour outside the
                    lump, or on the wall) has B compression >= seal_compression
  collapsed_cores   6-connected regions of boxes with B compression > collapse_compression
  max_density       the largest A- plus B-energy in one box (box volume 1)
  max_compression   the largest B compression b / b_rest
  edge_share        share of all energy in the outermost layer of boxes (the wall is near)
  steps, substeps   from the grid
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from physics.fluid.grid import QUANTUM, Grid
from physics.fluid.settings import Settings


def stats(grid: Grid, settings: Settings | None = None) -> dict:
    """Energy, drift, occupied size, lumps, sealed lumps, collapsed cores, densities."""
    settings = settings or Settings.all_on()
    a_quanta = grid.a_energy.numpy()
    b_quanta = grid.b_energy.numpy()
    total_quanta = int(a_quanta.sum()) + int(b_quanta.sum())
    a = np.maximum(a_quanta, 0) * QUANTUM
    b = np.maximum(b_quanta, 0) * QUANTUM
    energy = a + b
    compression = b / settings.b_rest
    occupied = int((energy > settings.occupied_floor * grid.start_a_mean).sum())
    lump_label, lumps = ndimage.label(a >= settings.lump_factor * grid.start_a_mean)
    _, cores = ndimage.label(compression > settings.collapse_compression)
    inner = ndimage.binary_erosion(lump_label > 0, border_value=0)
    surface_open = (lump_label > 0) & ~inner & (compression < settings.seal_compression)
    open_lumps = np.unique(lump_label[surface_open])
    edge = energy.copy()
    edge[1:-1, 1:-1, 1:-1] = 0.0
    return {"total_energy": total_quanta * QUANTUM,
            "energy_drift": total_quanta - grid.start_total,
            "occupied": occupied,
            "size_vs_start": occupied / max(grid.start_occupied, 1),
            "lumps": int(lumps),
            "sealed_lumps": int(lumps - open_lumps.size),
            "collapsed_cores": int(cores),
            "max_density": float(energy.max()),
            "max_compression": float(compression.max()),
            "edge_share": float(edge.sum() / max(energy.sum(), 1e-300)),
            "steps": grid.steps,
            "substeps": grid.substeps}
