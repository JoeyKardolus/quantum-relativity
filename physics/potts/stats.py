"""Stats: what the universe holds now, beside where it started (host numbers).

Contract: `stats(universe, settings=None)` refreshes the cell sums (cell_sums.py) and
returns a dict of plain Python numbers:
  total_energy      total energy (quanta * QUANTUM); energy_drift its change in quanta
                    since the start (0: energy is exact)
  cells, joints     living energy volumes (A) and joints (B); cell_count_growth their
                    number over the start's (only a split could raise it; none is built)
  collapses         joints that have collapsed since the start
  occupied          boxes held by a cell; occupied_growth occupied / start occupied
                    (space grows as cells claim empty boxes at the edge)
  radius            root mean square distance of the occupied boxes from the grid's
                    centre, in boxes
  lumps             energy volumes whose energy per box is at least lump_factor times
                    the start's mean A per box; sealed_lumps those of them whose every
                    joint is squeezed to the seal or past it (nothing leaks out)
  abab_breaks       touching pairs of different cells of the same kind (0 under ABAB)
  edge_share        share of the energy in the grid's outermost boxes (not ~0: the run
                    has reached the closed edge and is no longer open space)
  clock_min, clock_mean  over living cells; joint_compression_max, _mean over joints
  moves             boxes that changed owner in the last step
`settings` defaults to Settings.all_on().
"""
from __future__ import annotations

import numpy as np

from physics.potts.cell_sums import cell_sums
from physics.potts.contact_pairs import contact_pairs
from physics.potts.settings import Settings
from physics.potts.universe import ENERGY_VOLUME, JOINT, QUANTUM, Universe


def stats(universe: Universe, settings: Settings | None = None) -> dict:
    """Energy, cells and joints, collapses, occupied space, lumps and seals, ABAB, edge, clocks."""
    settings = Settings.all_on() if settings is None else settings
    cell_sums(universe, settings)
    owner = universe.owner.numpy()
    energy = universe.energy.numpy()
    kind = universe.kind.numpy()
    alive = universe.alive.numpy() == 1
    boxes = universe.cell_boxes.numpy()
    held = universe.cell_energy.numpy() * QUANTUM
    compression = universe.cell_compression.numpy()
    clock = universe.cell_clock.numpy()
    total = int(energy.sum())
    occupied = owner >= 0
    side = universe.side
    centre = 0.5 * (side - 1)
    places = np.argwhere(occupied).astype(np.float64) - centre
    edge = np.zeros_like(occupied)
    edge[[0, -1], :, :] = edge[:, [0, -1], :] = edge[:, :, [0, -1]] = True
    living = alive & (boxes > 0)
    volumes = living & (kind == ENERGY_VOLUME)
    joints = living & (kind == JOINT)
    low, high, _ = contact_pairs(owner)
    lumps = np.flatnonzero(volumes & (held >= settings.lump_factor * universe.rest_a * np.maximum(boxes, 1)))
    sealed = sum(1 for lump in lumps if _sealed(lump, low, high, kind, compression, settings.seal_compression))
    return {
        "total_energy": total * QUANTUM,
        "energy_drift": total - universe.start_total,
        "cells": int(volumes.sum()),
        "joints": int(joints.sum()),
        "cell_count_growth": float((volumes.sum() + joints.sum()) / (universe.start_cells + universe.start_joints)),
        "collapses": int(universe.collapses),
        "occupied": int(occupied.sum()),
        "occupied_growth": float(occupied.sum() / universe.start_occupied),
        "radius": float(np.sqrt((places ** 2).sum(axis=1).mean())) if places.size else 0.0,
        "lumps": int(lumps.size),
        "sealed_lumps": int(sealed),
        "abab_breaks": int(np.sum(kind[low] == kind[high])),
        "edge_share": float(energy[edge].sum() / total) if total else 0.0,
        "clock_min": float(clock[living].min()) if living.any() else 1.0,
        "clock_mean": float(clock[living].mean()) if living.any() else 1.0,
        "joint_compression_max": float(compression[joints].max()) if joints.any() else 0.0,
        "joint_compression_mean": float(compression[joints].mean()) if joints.any() else 0.0,
        "moves": int(universe.last_moves),
    }


def _sealed(lump: int, low: np.ndarray, high: np.ndarray, kind: np.ndarray, compression: np.ndarray,
            seal: float) -> bool:
    around = np.concatenate([high[low == lump], low[high == lump]])
    around = around[kind[around] == JOINT]
    return bool(around.size) and bool(np.all(compression[around] >= seal))
