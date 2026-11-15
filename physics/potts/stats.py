"""Stats: what the universe holds now, beside where it started (host numbers).

Contract: `stats(universe, settings=None)` refreshes the cell sums (cell_sums.py) and
returns a dict of plain Python numbers (box arrays are read over the active region,
region_host.py; outside it every box is empty):
  total_energy      total energy, the sum of |psi|**2 over the boxes; energy_drift its
                    change since the start (energy units), relative_drift that over
                    the start's total (the hops and turns are exact rotations, so
                    only float64 rounding moves it)
  fluid_energy      the energy fluid's own energy in its equation (fluid_energy.py):
                    flow, pull and push terms; not conserved while the pull, the
                    cells and friction change (reported, not a rule)
  heat              total heat (energy units): directed flow turned into jiggling by
                    friction at A-B faces; bookkeeping, not part of total_energy
  cells, joints     living energy volumes (A) and joints (B); cell_count_growth their
                    number over the start's (volume splits raise it, collapses lower it)
  collapses, splits joints that have collapsed and volumes that have torn since the start
  occupied          boxes held by a cell; occupied_growth occupied / start occupied
                    (space grows as cells claim empty boxes at the edge)
  radius            root mean square distance of the occupied boxes from the grid's
                    centre, in boxes
  lumps             energy volumes whose energy per box is at least lump_factor times
                    the start's mean A per box; sealed_lumps those of them whose every
                    joint is squeezed to the seal or past it (nothing leaks out)
  wrap_breaks       pairs of energy volumes that touch (face, edge or corner,
                    touch_pairs.py) plus energy volumes touching empty space (0 while
                    every energy volume is fully wrapped in joints)
  web_pieces        how many face-connected pieces all joint boxes together make (1:
                    the joints form one web)
  edge_share        share of the energy in the grid's outermost boxes (not ~0: the run
                    has reached the closed edge and is no longer open space)
  clock_min, clock_mean  over living cells; joint_compression_max, _mean over joints
  size_ratio        mean box count of living joints over that of living energy volumes
  moves             boxes that changed owner in the last step
  substeps, box_updates  finest substeps and box updates since the start
`settings` defaults to Settings.all_on().
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from physics.potts.cell_sums import cell_sums
from physics.potts.contact_pairs import contact_pairs
from physics.potts.fluid_energy import fluid_energy
from physics.potts.region_host import region_host
from physics.potts.settings import Settings
from physics.potts.touch_pairs import touch_pairs
from physics.potts.universe import ENERGY_VOLUME, JOINT, Universe


def stats(universe: Universe, settings: Settings | None = None) -> dict:
    """Energy, heat, cells and joints, collapses and splits, space, lumps and seals, wrapping, edge, clocks."""
    settings = Settings.all_on() if settings is None else settings
    cell_sums(universe, settings)
    owner = region_host(universe, universe.owner)
    psi = region_host(universe, universe.psi)
    energy = (psi.astype(np.float64) ** 2).sum(axis=-1)
    heat = region_host(universe, universe.heat)
    kind = universe.kind.numpy()
    alive = universe.alive.numpy() == 1
    boxes = universe.cell_boxes.numpy()
    held = universe.cell_energy.numpy()
    compression = universe.cell_compression.numpy()
    clock = universe.cell_clock.numpy()
    total = float(energy.sum())
    occupied = owner >= 0
    side = universe.side
    centre = 0.5 * (side - 1)
    places = np.argwhere(occupied).astype(np.float64) + np.array(universe.low) - centre
    edge = np.zeros_like(occupied)
    for axis in range(3):
        moved = np.moveaxis(edge, axis, 0)
        if universe.low[axis] == 0:
            moved[0] = True
        if universe.high[axis] == side:
            moved[-1] = True
    living = alive & (boxes > 0)
    volumes = living & (kind == ENERGY_VOLUME)
    joints = living & (kind == JOINT)
    low, high, _ = contact_pairs(owner)
    touch_low, touch_high = touch_pairs(owner)
    a_high = kind[touch_high] == ENERGY_VOLUME
    a_low = (touch_low >= 0) & (kind[np.maximum(touch_low, 0)] == ENERGY_VOLUME)
    wrap_breaks = int(np.sum(a_high & (a_low | (touch_low < 0))))
    joint_boxes = occupied & (kind[np.maximum(owner, 0)] == JOINT)
    lumps = np.flatnonzero(volumes & (held >= settings.lump_factor * universe.rest_a * np.maximum(boxes, 1)))
    sealed = sum(1 for lump in lumps if _sealed(lump, low, high, kind, compression, settings.seal_compression))
    return {
        "total_energy": total,
        "energy_drift": total - universe.start_total,
        "relative_drift": (total - universe.start_total) / universe.start_total if universe.start_total else 0.0,
        "fluid_energy": fluid_energy(universe, settings),
        "heat": float(heat.sum(dtype=np.float64)),
        "cells": int(volumes.sum()),
        "joints": int(joints.sum()),
        "cell_count_growth": float((volumes.sum() + joints.sum()) / (universe.start_cells + universe.start_joints)),
        "collapses": int(universe.collapses),
        "splits": int(universe.splits),
        "occupied": int(occupied.sum()),
        "occupied_growth": float(occupied.sum() / universe.start_occupied),
        "radius": float(np.sqrt((places ** 2).sum(axis=1).mean())) if places.size else 0.0,
        "lumps": int(lumps.size),
        "sealed_lumps": int(sealed),
        "wrap_breaks": wrap_breaks,
        "web_pieces": int(ndimage.label(joint_boxes)[1]),
        "edge_share": float(energy[edge].sum() / total) if total else 0.0,
        "clock_min": float(clock[living].min()) if living.any() else 1.0,
        "clock_mean": float(clock[living].mean()) if living.any() else 1.0,
        "joint_compression_max": float(compression[joints].max()) if joints.any() else 0.0,
        "joint_compression_mean": float(compression[joints].mean()) if joints.any() else 0.0,
        "size_ratio": float(boxes[joints].mean() / boxes[volumes].mean()) if joints.any() and volumes.any() else 0.0,
        "moves": int(universe.last_moves),
        "substeps": int(universe.substeps),
        "box_updates": int(universe.box_updates),
    }


def _sealed(lump: int, low: np.ndarray, high: np.ndarray, kind: np.ndarray, compression: np.ndarray,
            seal: float) -> bool:
    around = np.concatenate([high[low == lump], low[high == lump]])
    around = around[kind[around] == JOINT]
    return bool(around.size) and bool(np.all(compression[around] >= seal))
