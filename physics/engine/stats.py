"""Stats: what one universe holds now, beside where it started (host numbers).

Contract: `stats(state, settings=None, universe=0)` copies that universe's boxes once
(host_boxes.py), reads its cells (cells.py) and returns a dict of plain Python numbers:
  amount, plus, minus  the whole quanta ignoring sign, the plus (pulling) and the minus
                    (pushing) quanta, exact ints
  energy_drift      amount - the start's amount, in quanta (0: exact, bible rule 4);
                    relative_drift that over the start's amount
  plus_share        plus / amount (collapse is the one rule that turns minus into plus)
  cells, joints     energy volumes and joints of the cell graph (cells.py);
                    cell_count_growth their number over the start's (start["cells"], set
                    by new_universe; NaN when unknown)
  seam_faces, seam_pairs  faces where two joints meet and how many joint pairs meet (seams.py)
  collapses, collapsed_quanta, crossed, rehomes  joints collapsed and the minus quanta
                    they turned plus, quanta that crossed joints and boxes that changed
                    cell since the start (state tallies, state.collapses)
  splits, torn      volume splits so far and the plus quanta their tears turned minus
                    (volume_split.py)
  occupied          boxes holding space; occupied_growth occupied / the start's
  radius            root mean square distance of the occupied boxes from the cube's
                    centre, in boxes
  lumps             energy volumes holding at least LUMP_FACTOR (2) times the start's
                    mean plus quanta per plus box, per box; sealed_lumps those whose every
                    joint is squeezed to the seal or past it (settings.seal_squeeze:
                    nothing crosses)
  wrap_breaks       energy volumes with a box that has an empty box or a wall among its
                    26 neighbours (0 while every volume is fully wrapped; exchange.py
                    keeps them wrapped)
  web_pieces        face-connected pieces of all joint boxes together, joined round
                    the edges of a wrapping universe (1: one web)
  edge_share        share of the amount in the cube's outermost boxes (not ~0 in a
                    universe with walls: the run reached them; a wrapping universe has
                    no walls)
  clock_min, clock_mean  clock rate over the occupied boxes, each its cell's
                    (cell_laws.py: 1 / (1 + packing / clock_packing))
  local_time_min, local_time_mean, local_time_max  the cells' mean clock readings, the
                    proper time they lived
  squeeze_max, squeeze_mean  joints' squeeze, wanted length / length - 1, the
                    wanted length (size_energy[1] / quanta) ** 3 cubed rooted
  volume_across     mean over energy volumes of their box count's cube root
`settings` defaults to Settings(). Raises ValueError for a universe the batch lacks.
"""
from __future__ import annotations

import numpy as np

from physics.engine.cells import ENERGY_VOLUME, JOINT, cells
from physics.engine.group_label import group_label
from physics.engine.host_boxes import host_boxes
from physics.engine.near import near
from physics.engine.seams import seams
from physics.engine.settings import Settings
from physics.engine.state import State

LUMP_FACTOR = 2.0


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
    edge = np.zeros(quanta.shape, dtype=bool) if state.wrap else np.ones(quanta.shape, dtype=bool)
    edge[1:-1, 1:-1, 1:-1] = False
    volumes = found["kind"] == ENERGY_VOLUME
    joint_boxes = quanta < 0
    squeeze = _squeeze(found, boxes["size_energy"][1])
    rate = np.ones(found["kind"].size)
    if settings.clock_packing > 0:
        rate = 1.0 / (1.0 + found["quanta"] / np.maximum(found["boxes"], 1) / settings.clock_packing)
    rate = rate[found["owner"][occupied]]
    crossed, collapsed, rehomes, collapsed_quanta = (int(value) for value in state.tallies.numpy()[universe])
    collapsed = int(state.collapses[universe])
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
        "collapses": collapsed, "collapsed_quanta": collapsed_quanta, "crossed": crossed, "rehomes": rehomes,
        "splits": int(state.splits[universe]), "torn": int(state.torn[universe]),
        "occupied": int(occupied.sum()),
        "occupied_growth": float(occupied.sum() / start["boxes"]) if start["boxes"] else 0.0,
        "radius": float(np.sqrt((places ** 2).sum(axis=1).mean())) if places.size else 0.0,
        "lumps": int(lumps.size),
        "sealed_lumps": sum(1 for lump in lumps if _sealed(found, lump, squeeze, settings)),
        "wrap_breaks": _wrap_breaks(found["owner"], quanta, state.wrap),
        "web_pieces": group_label(joint_boxes, state.wrap)[1],
        "edge_share": float(np.abs(quanta[edge]).sum() / amount) if amount else 0.0,
        "clock_min": float(rate.min()) if rate.size else 1.0,
        "clock_mean": float(rate.mean()) if rate.size else 1.0,
        "local_time_min": float(found["clock"].min()) if found["clock"].size else 0.0,
        "local_time_mean": float(found["clock"].mean()) if found["clock"].size else 0.0,
        "local_time_max": float(found["clock"].max()) if found["clock"].size else 0.0,
        "squeeze_max": float(np.nanmax(squeeze)) if (~np.isnan(squeeze)).any() else 0.0,
        "squeeze_mean": float(np.nanmean(squeeze)) if (~np.isnan(squeeze)).any() else 0.0,
        "volume_across": float(np.cbrt(found["boxes"][volumes]).mean()) if volumes.any() else 0.0,
    }


def _lumps(found: dict, start: dict) -> np.ndarray:
    """Energy volumes packed at least LUMP_FACTOR times the start's mean plus quanta per plus box."""
    rest = start["plus"] / start["plus_boxes"] if start["plus_boxes"] else 0.0
    count = found["volumes"]
    packed = found["quanta"][:count] >= LUMP_FACTOR * rest * np.maximum(found["boxes"][:count], 1)
    return np.flatnonzero(packed)


def _squeeze(found: dict, constant: float) -> np.ndarray:
    """Each joint's squeeze, (wanted / size) ** (1/3) - 1; NaN for a volume."""
    wanted = (constant / np.maximum(found["quanta"], 1).astype(np.float64)) ** 3
    joints = found["kind"] == JOINT
    return np.where(joints, np.cbrt(wanted / np.maximum(found["size"], 1e-6)) - 1.0, np.nan)


def _sealed(found: dict, lump: int, squeeze: np.ndarray, settings: Settings) -> bool:
    """True when every joint of the volume is squeezed to the seal or past it."""
    around = np.flatnonzero((found["between"] == lump).any(axis=1))
    return bool(around.size) and bool(np.all(squeeze[around] >= settings.seal_squeeze))


def _wrap_breaks(owner: np.ndarray, quanta: np.ndarray, wrap: bool) -> int:
    """Energy volumes with a box that has an empty box or a wall among its 26 neighbours."""
    touched = owner[(quanta > 0) & near(quanta == 0, wrap, wall=True)]
    return int(np.unique(touched).size)
