"""Cells: the energy volumes and joints read off the signed quanta, on the host (the engine keeps no labels).

Contract: `cells(boxes)` takes host_boxes() of one universe (quanta, clock; host_boxes.py)
and returns a dict of numpy arrays describing its cells:
  owner     int32 (side, side, side): the cell each box belongs to, -1 for an empty box
  kind      int32 per cell: ENERGY_VOLUME (0) or JOINT (1)
  boxes     int64 per cell: its box count, its size
  quanta    int64 per cell: its whole quanta (amount, ignoring sign)
  centre    float64 (cells, 3): the mean index of its boxes
  clock     float64 per cell: the mean local clock reading of its boxes, the proper time
            it has lived (tick_clocks.py)
  volumes   int: how many energy volumes; they are cells 0 .. volumes - 1, ordered by
            their lowest box index, joints after them
An energy volume is a group of plus boxes connected across faces, edges or corners (26
neighbours; the bible keeps energy volumes apart on all 26 sides, so two volumes that
touch are one). A joint is a group of minus boxes connected across faces; the joints
normally form one web (bible "The foundation"), so most universes hold one joint cell.
The labels are read fresh each call and are not kept: a cell's id can change from one
call to the next (clocked_run follows volumes by their shared boxes).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

ENERGY_VOLUME = 0
JOINT = 1
_CORNERS = np.ones((3, 3, 3), dtype=bool)


def cells(boxes: dict) -> dict:
    """Owner per box and kind, size, quanta, centre and mean clock per cell."""
    quanta = boxes["quanta"]
    volume_ids, volumes = ndimage.label(quanta > 0, structure=_CORNERS)
    joint_ids, joints = ndimage.label(quanta < 0)
    owner = np.where(volume_ids > 0, volume_ids - 1, np.where(joint_ids > 0, joint_ids - 1 + volumes, -1))
    owner = owner.astype(np.int32)
    count = volumes + joints
    held = owner.ravel()
    occupied = held >= 0
    ids = held[occupied]
    box_counts = np.bincount(ids, minlength=count).astype(np.int64)
    amounts = np.bincount(ids, weights=np.abs(quanta.ravel()[occupied]).astype(np.float64), minlength=count)
    clocks = np.bincount(ids, weights=boxes["clock"].ravel()[occupied].astype(np.float64), minlength=count)
    places = np.argwhere(owner >= 0)
    order_ids = owner[owner >= 0]
    centre = np.stack([np.bincount(order_ids, weights=places[:, axis].astype(np.float64), minlength=count)
                       for axis in range(3)], axis=1)
    sizes = np.maximum(box_counts, 1)
    kind = np.where(np.arange(count) < volumes, ENERGY_VOLUME, JOINT).astype(np.int32)
    return {"owner": owner, "kind": kind, "boxes": box_counts, "quanta": np.rint(amounts).astype(np.int64),
            "centre": centre / sizes[:, None], "clock": clocks / sizes, "volumes": int(volumes)}
