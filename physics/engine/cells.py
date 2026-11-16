"""Cells: the energy volumes and joints read off the signed quanta, on the host (the engine keeps no labels).

Contract: `cells(boxes)` takes host_boxes() of one universe (quanta, clock; host_boxes.py)
and returns a dict of numpy arrays describing its cells:
  owner     int32 (side, side, side): the cell each box belongs to, -1 for an empty box
  kind      int32 per cell: ENERGY_VOLUME (0) or JOINT (1)
  wraps     int32 per cell: for a joint the energy volume it wraps, -1 for an energy
            volume and for a joint with no energy volume in its universe
  boxes     int64 per cell: its box count, its size
  quanta    int64 per cell: its whole quanta (amount, ignoring sign)
  centre    float64 (cells, 3): the mean index of its boxes
  clock     float64 per cell: the mean local clock reading of its boxes, the proper time
            it has lived (tick_clocks.py)
  volumes   int: how many energy volumes; they are cells 0 .. volumes - 1, ordered by
            their lowest box index, joints after them, ordered the same way
An energy volume is a group of plus boxes connected across faces, edges or corners (26
neighbours; the bible keeps energy volumes apart on all 26 sides, so two volumes that
touch are one). The minus boxes together are the joint web; a joint is the part of the
web that wraps one energy volume: the energy volumes grow into the web together, one
face step at a time through minus boxes only, every minus box going to the volume that
reaches it first (two at once: the higher id), and a joint is a face-connected group of
minus boxes reached by one volume. So every joint is a connected region of one sign,
every energy volume is wrapped by its own joint, and the joints meet at seams
(seams.py), the mortar's middle between two volumes. Minus boxes no volume reaches
(a piece of web with no energy volume) form joints of their own, one per face-connected
group, wrapping none. The labels are read fresh each call and are not kept: a cell's id can change
from one call to the next (clocked_run follows volumes by their shared boxes).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ENERGY_VOLUME = 0
JOINT = 1
_CORNERS = np.ones((3, 3, 3), dtype=bool)
_FACES = ndimage.generate_binary_structure(3, 1)


def cells(boxes: dict) -> dict:
    """Owner per box and kind, wrapped volume, size, quanta, centre and mean clock per cell."""
    quanta = boxes["quanta"]
    volume_ids, volumes = ndimage.label(quanta > 0, structure=_CORNERS)
    nearest = _nearest_volume(volume_ids, quanta < 0, volumes)
    joint_ids, joint_wraps = _joints(quanta < 0, nearest)
    joints = joint_wraps.size
    owner = np.where(volume_ids > 0, volume_ids - 1, np.where(joint_ids >= 0, joint_ids + volumes, -1))
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
    wraps = np.concatenate([np.full(volumes, -1, dtype=np.int32), joint_wraps.astype(np.int32)])
    return {"owner": owner, "kind": kind, "wraps": wraps, "boxes": box_counts,
            "quanta": np.rint(amounts).astype(np.int64), "centre": centre / sizes[:, None], "clock": clocks / sizes,
            "volumes": int(volumes)}


def _nearest_volume(volume_ids: np.ndarray, minus: np.ndarray, volumes: int) -> np.ndarray:
    """Per minus box the 0-based energy volume that reaches it first through the web, -1 where none reaches."""
    nearest = np.full(volume_ids.shape, -1, dtype=np.int64)
    if volumes == 0 or not minus.any():
        return nearest
    occupied = (volume_ids > 0) | minus
    low = np.argwhere(occupied).min(axis=0)
    high = np.argwhere(occupied).max(axis=0) + 1
    window = tuple(slice(start, stop) for start, stop in zip(low, high))
    labels = volume_ids[window].astype(np.int64)
    web = minus[window]
    while True:
        grown = ndimage.grey_dilation(labels, footprint=_FACES, mode="constant", cval=0)
        reached = web & (labels == 0) & (grown > 0)
        if not reached.any():
            break
        labels[reached] = grown[reached]
    nearest[window] = np.where(web, labels - 1, -1)
    return nearest


def _joints(minus: np.ndarray, nearest: np.ndarray) -> tuple:
    """Joint id per box (-1 off the web) and per joint the volume it wraps: face-connected minus boxes of one volume."""
    flat = np.flatnonzero(minus.ravel())
    ids = np.full(minus.size, -1, dtype=np.int64)
    if flat.size == 0:
        return ids.reshape(minus.shape), np.zeros(0, dtype=np.int64)
    index_of = np.full(minus.size, -1, dtype=np.int64)
    index_of[flat] = np.arange(flat.size)
    wrapped = nearest.ravel()
    starts, ends = [], []
    for axis in range(3):
        stride = int(np.prod(minus.shape[axis + 1:]))
        last = (np.unravel_index(flat, minus.shape)[axis] == minus.shape[axis] - 1)
        candidates = flat[~last]
        neighbours = candidates + stride
        joined = minus.ravel()[neighbours] & (wrapped[candidates] == wrapped[neighbours])
        starts.append(index_of[candidates[joined]])
        ends.append(index_of[neighbours[joined]])
    starts, ends = np.concatenate(starts), np.concatenate(ends)
    graph = coo_matrix((np.ones(starts.size, dtype=np.int8), (starts, ends)), shape=(flat.size, flat.size))
    _, components = connected_components(graph, directed=False)
    order = np.unique(components, return_index=True)[1]
    renumber = np.empty(order.size, dtype=np.int64)
    renumber[np.argsort(order)] = np.arange(order.size)
    joint_of = renumber[components]
    ids[flat] = joint_of
    wraps = np.full(order.size, -1, dtype=np.int64)
    wraps[joint_of] = wrapped[flat]
    return ids.reshape(minus.shape), wraps
