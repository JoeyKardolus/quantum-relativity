"""Cells: the energy volumes and joints of one universe's cell graph, read off the boxes, on the host.

Contract: `cells(boxes)` takes host_boxes() of one universe and returns a dict of numpy
arrays describing its cells that exist:
  owner     int32 (side, side, side): the cell each box belongs to (numbered below), -1
            for empty space
  kind      int32 per cell: ENERGY_VOLUME (0) or JOINT (1)
  between   int32 (cells, 2): for a joint the two energy volumes it lies between, lower
            number first, -1 for a side facing empty space or a wall; (-1, -1) for a
            volume
  place     int64 (cells, 3): the cell's start lattice place (new_universe), -1 for a
            cell made later or read off the boxes
  id        int64 per cell: its id in the state's cell table
  boxes     int64 per cell: its box count, its size in Planck boxes
  quanta    int64 per cell: its whole quanta (amount, ignoring sign)
  size      float64 per cell: the size its rule gives it (state.size)
  centre    float64 (cells, 3): the mean index of its boxes
  clock     float64 per cell: its own clock, the proper time it has lived
  volumes   int: how many energy volumes; they are cells 0 .. volumes - 1, joints after
            them, each in id order
Two neighbouring volumes share exactly one joint and every volume is wrapped on all 26
sides (the engine keeps both, exchange.py); the numbering is the same at every call
while no cell is made or lost.
"""
from __future__ import annotations

import numpy as np

ENERGY_VOLUME = 0
JOINT = 1


def cells(boxes: dict) -> dict:
    """Owner per box and kind, the volumes each joint lies between, place, size, quanta, centre and clock per cell."""
    kind = np.asarray(boxes["kind"])
    order = np.concatenate([np.flatnonzero(kind == 1), np.flatnonzero(kind == -1)])
    renumber = np.full(kind.size + 1, -1, dtype=np.int64)
    renumber[order] = np.arange(order.size)
    owner = renumber[boxes["owner"]].astype(np.int32)
    volumes = int((kind == 1).sum())
    sides = np.asarray(boxes["between"])[order]
    between = np.where(sides >= 0, renumber[np.maximum(sides, 0)], -1).astype(np.int32)
    between[:volumes] = -1
    found = _tallies(owner, boxes["quanta"], order.size)
    return {"owner": owner, "kind": np.where(np.arange(order.size) < volumes, ENERGY_VOLUME, JOINT).astype(np.int32),
            "between": between, "place": np.asarray(boxes["place"])[order].astype(np.int64),
            "id": order.astype(np.int64), "size": np.asarray(boxes["size"], dtype=np.float64)[order],
            "clock": np.asarray(boxes["cell_clock"], dtype=np.float64)[order], **found, "volumes": volumes}


def _tallies(owner: np.ndarray, quanta: np.ndarray, count: int) -> dict:
    """Box count, quanta and box-index centre per cell."""
    held = owner.ravel()
    occupied = held >= 0
    ids = held[occupied]
    box_counts = np.bincount(ids, minlength=count).astype(np.int64)
    amounts = np.bincount(ids, weights=np.abs(quanta.ravel()[occupied]).astype(np.float64), minlength=count)
    places = np.argwhere(owner >= 0)
    centre = np.stack([np.bincount(ids, weights=places[:, axis].astype(np.float64), minlength=count)
                       for axis in range(3)], axis=1)
    sizes = np.maximum(box_counts, 1)
    return {"boxes": box_counts, "quanta": np.rint(amounts).astype(np.int64), "centre": centre / sizes[:, None]}
