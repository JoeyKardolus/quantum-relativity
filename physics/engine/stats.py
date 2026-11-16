"""Stats: what one universe holds now, beside where it started (host numbers).

Contract: `stats(state, settings=None, universe=0)` copies that universe's boxes once
(host_boxes.py), reads its cells (cells.py) and returns a dict of plain Python numbers:
  amount, plus, minus  the whole quanta ignoring sign, the plus (pulling) and the minus
                    (pushing) quanta, exact ints
  energy_drift      amount - the start's amount, in quanta (0: exact, bible rule 4);
                    relative_drift that over the start's amount
  plus_share        plus / amount (collapse is the one rule that turns minus into plus)
  cells, joints     energy volumes and joint cells (cells.py: a joint wraps one volume);
                    cell_count_growth their number over the start's (start["cells"], set
                    by new_universe; NaN when unknown)
  seam_faces, seam_pairs  faces where two joints meet and how many joint pairs meet (seams.py)
  collapses, collapsed_quanta, spread, claims  joint boxes collapsed and the minus quanta
                    they turned plus, quanta spread by drift and claims made since the
                    start (state tallies)
  occupied          boxes holding space; occupied_growth occupied / the start's
  radius            root mean square distance of the occupied boxes from the cube's
                    centre, in boxes
  lumps             energy volumes holding at least LUMP_FACTOR (2) times the start's
                    mean plus quanta per plus box, per box; sealed_lumps those whose every
                    joint box on a face of theirs is squeezed to the seal or past it
                    (settings.seal_squeeze: nothing passes)
  wrap_breaks       energy volumes with a box that has an empty box or a wall among its
                    26 neighbours (0 while every volume is fully wrapped; two volumes
                    never touch, cells.py joins them)
  web_pieces        face-connected pieces of all joint boxes together (1: one web)
  edge_share        share of the amount in the cube's outermost boxes (not ~0: the run
                    reached the closed walls and is no longer open space)
  clock_min, clock_mean  clock rate over the occupied boxes (clock_rate.py: 1 / (1 +
                    |quanta| / clock_packing); 1 everywhere when local clocks are off)
  local_time_min, local_time_mean, local_time_max  the cells' mean clock readings, the
                    proper time they lived
  squeeze_max, squeeze_mean  joint boxes' squeeze, |quanta| / joint_rest - 1
  volume_across     mean over energy volumes of their box count's cube root
`settings` defaults to Settings(). Raises ValueError for a universe the batch lacks.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from physics.engine.cells import ENERGY_VOLUME, cells
from physics.engine.host_boxes import host_boxes
from physics.engine.seams import seams
from physics.engine.settings import Settings
from physics.engine.state import State

LUMP_FACTOR = 2.0
_CORNERS = np.ones((3, 3, 3), dtype=bool)


def stats(state: State, settings: Settings | None = None, universe: int = 0) -> dict:
    """Amounts, drift, cells, seams, lumps, seals, wrapping, web, edge, space, clocks of one universe."""
    settings = Settings() if settings is None else settings
    boxes = host_boxes(state, universe)
    quanta = boxes["quanta"].astype(np.int64)
    found = cells(boxes)
    start = state.start[universe]
    plus = int(quanta[quanta > 0].sum())
    minus = int(-quanta[quanta < 0].sum())
    amount = plus + minus
    occupied = quanta != 0
    side = state.side
    places = np.argwhere(occupied).astype(np.float64) - 0.5 * (side - 1)
    edge = np.ones(quanta.shape, dtype=bool)
    edge[1:-1, 1:-1, 1:-1] = False
    volumes = found["kind"] == ENERGY_VOLUME
    joint_boxes = quanta < 0
    squeeze = np.abs(quanta[joint_boxes]) / settings.joint_rest - 1.0
    rate = 1.0 / (1.0 + np.abs(quanta[occupied]) / settings.clock_packing) if settings.clock_packing > 0 else \
        np.ones(int(occupied.sum()))
    spread, collapsed, claims, collapsed_quanta = (int(value) for value in state.tallies.numpy()[universe])
    lumps = _lumps(found, start)
    meeting = seams(found)
    return {
        "amount": amount, "plus": plus, "minus": minus,
        "energy_drift": amount - start["amount"],
        "relative_drift": (amount - start["amount"]) / start["amount"] if start["amount"] else 0.0,
        "plus_share": plus / amount if amount else 0.0,
        "cells": int(volumes.sum()), "joints": int((~volumes).sum()),
        "cell_count_growth": found["kind"].size / start["cells"] if start.get("cells") else float("nan"),
        "seam_faces": meeting["faces"], "seam_pairs": int(len(meeting["pairs"])),
        "collapses": collapsed, "collapsed_quanta": collapsed_quanta, "spread": spread, "claims": claims,
        "occupied": int(occupied.sum()),
        "occupied_growth": float(occupied.sum() / start["boxes"]) if start["boxes"] else 0.0,
        "radius": float(np.sqrt((places ** 2).sum(axis=1).mean())) if places.size else 0.0,
        "lumps": int(lumps.size),
        "sealed_lumps": sum(1 for lump in lumps if _sealed(found["owner"] == lump, quanta, settings)),
        "wrap_breaks": _wrap_breaks(found["owner"], quanta),
        "web_pieces": int(ndimage.label(joint_boxes)[1]),
        "edge_share": float(np.abs(quanta[edge]).sum() / amount) if amount else 0.0,
        "clock_min": float(rate.min()) if rate.size else 1.0,
        "clock_mean": float(rate.mean()) if rate.size else 1.0,
        "local_time_min": float(found["clock"].min()) if found["clock"].size else 0.0,
        "local_time_mean": float(found["clock"].mean()) if found["clock"].size else 0.0,
        "local_time_max": float(found["clock"].max()) if found["clock"].size else 0.0,
        "squeeze_max": float(squeeze.max()) if squeeze.size else 0.0,
        "squeeze_mean": float(squeeze.mean()) if squeeze.size else 0.0,
        "volume_across": float(np.cbrt(found["boxes"][volumes]).mean()) if volumes.any() else 0.0,
    }


def _lumps(found: dict, start: dict) -> np.ndarray:
    """Energy volumes packed at least LUMP_FACTOR times the start's mean plus quanta per plus box."""
    rest = start["plus"] / start["plus_boxes"] if start["plus_boxes"] else 0.0
    count = found["volumes"]
    packed = found["quanta"][:count] >= LUMP_FACTOR * rest * np.maximum(found["boxes"][:count], 1)
    return np.flatnonzero(packed)


def _sealed(volume: np.ndarray, quanta: np.ndarray, settings: Settings) -> bool:
    """True when every joint box on a face of the volume is squeezed to the seal or past it."""
    around = ndimage.binary_dilation(volume) & (quanta < 0)
    if not around.any():
        return False
    return bool(np.all(np.abs(quanta[around]) / settings.joint_rest - 1.0 >= settings.seal_squeeze))


def _wrap_breaks(owner: np.ndarray, quanta: np.ndarray) -> int:
    """Energy volumes with a box that has an empty box or a wall among its 26 neighbours."""
    open_space = np.pad(quanta == 0, 1, constant_values=True)
    near_empty = ndimage.binary_dilation(open_space, structure=_CORNERS)[1:-1, 1:-1, 1:-1]
    touched = owner[(quanta > 0) & near_empty]
    return int(np.unique(touched).size)

