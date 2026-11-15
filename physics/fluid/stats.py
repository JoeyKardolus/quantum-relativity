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
  proper_volume     the occupied boxes' summed length cubed (local_length.py): space's size
  expansion         proper_volume / grid.start_proper_volume (space grows as lumps form)
  clock_slowest     the slowest clock rate in an occupied box (clock_rate.py, set here)
  clock_mean        the energy-weighted mean clock rate
  heat              the summed heat friction made (grid.heat)
  motion_energy     the summed directed-flow energy, (1/2) |p|**2 / e of both kinds
  pull_energy       (1/2) sum of A-energy times the pull potential (negative; the last solve)
  sealed_radius     the largest sealed region's equivalent-sphere radius (sealed_radius.py)
  steps, substeps   from the grid
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from physics.fluid.clock_rate import clock_rate
from physics.fluid.grid import QUANTUM, Grid
from physics.fluid.local_length import local_length
from physics.fluid.sealed_radius import sealed_radius
from physics.fluid.settings import Settings


def stats(grid: Grid, settings: Settings | None = None) -> dict:
    """Energy, drift, sizes, lumps, seals, cores, densities, clocks, heat, motion and pull energy."""
    settings = settings or Settings.all_on()
    a_quanta = grid.a_energy.numpy()
    b_quanta = grid.b_energy.numpy()
    total_quanta = int(a_quanta.sum()) + int(b_quanta.sum())
    a = np.maximum(a_quanta, 0) * QUANTUM
    b = np.maximum(b_quanta, 0) * QUANTUM
    energy = a + b
    compression = b / settings.b_rest
    space = energy > settings.occupied_floor * grid.start_a_mean
    occupied = int(space.sum())
    length = local_length(a, b, grid.start_a_mean, settings.b_rest)
    proper_volume = float((length[space] ** 3).sum())
    clock_rate(grid, settings)
    clock = grid.clock.numpy()
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
            "proper_volume": proper_volume,
            "expansion": proper_volume / max(grid.start_proper_volume, 1e-300),
            "clock_slowest": float(clock[space].min()) if occupied else 1.0,
            "clock_mean": float((clock * energy).sum() / max(energy.sum(), 1e-300)),
            "heat": float(grid.heat.numpy().astype(np.float64).sum()),
            "motion_energy": _motion(grid.a_momentum.numpy(), a) + _motion(grid.b_momentum.numpy(), b),
            "pull_energy": float(0.5 * (a * grid.potential.numpy()).sum()),
            "sealed_radius": sealed_radius(compression, length, settings.seal_compression),
            "steps": grid.steps,
            "substeps": grid.substeps}


def _motion(momentum: np.ndarray, energy: np.ndarray) -> float:
    squared = (momentum.astype(np.float64) ** 2).sum(axis=-1)
    return float(0.5 * np.divide(squared, energy, out=np.zeros_like(energy), where=energy > 1e-6).sum())
