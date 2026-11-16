"""Cells: the energy volumes and joints read off the signed quanta, on the host (the engine keeps no labels).

Contract: `cells(boxes)` takes host_boxes() of one universe (quanta, clock; host_boxes.py)
and returns a dict of numpy arrays describing its cells:
  owner     int32 (side, side, side): the cell each box belongs to, -1 for an empty box
  kind      int32 per cell: ENERGY_VOLUME (0) or JOINT (1)
  between   int32 (cells, 2): for a joint the two energy volumes it lies between, lower
            id first; -1 for a side that faces empty space (the outer skin) or no
            volume; (-1, -1) for an energy volume
  boxes     int64 per cell: its box count, its size
  quanta    int64 per cell: its whole quanta (amount, ignoring sign)
  centre    float64 (cells, 3): the mean index of its boxes
  clock     float64 per cell: the mean local clock reading of its boxes, the proper time
            it has lived (tick_clocks.py)
  volumes   int: how many energy volumes; they are cells 0 .. volumes - 1, ordered by
            their lowest box index, joints after them, ordered the same way
An energy volume is a group of plus boxes connected across faces, edges or corners (26
neighbours; the bible keeps energy volumes apart on all 26 sides, so two volumes that
touch are one). The minus boxes together are the joint web. Owner, 2026-11-16: there
is one shared joint between two neighbouring energy volumes (red | blue | red), not a
shell per volume, and the joints touch each other in one web. So the energy volumes and
empty space (empty boxes and the walls) grow into the web together, one face step at
a time through minus boxes only; every minus box keeps the first two different sources
that reach it (at a tie the higher volume id first, empty space after every volume), and
a joint is every minus box with the same two sources: the slab between two volumes, or
a volume's skin facing empty space. Counted in cells there is exactly one joint
between two neighbouring volumes, however many Planck boxes thick it is; the joints
meet at seams (seams.py), along the mortar's edges. Every energy volume is wrapped by
the joints around it. Where borders are ragged a joint can hold a few boxes apart from
its main slab (ties between near-equal sources); it is still that one joint. Minus
boxes no volume reaches form one joint between no volumes. The labels are read fresh each call and are not
kept: a cell's id can change from one call to the next (clocked_run follows volumes by
their shared boxes).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

ENERGY_VOLUME = 0
JOINT = 1
_CORNERS = np.ones((3, 3, 3), dtype=bool)
_FACES = ndimage.generate_binary_structure(3, 1)


def cells(boxes: dict) -> dict:
    """Owner per box and kind, the volumes each joint lies between, size, quanta, centre and mean clock per cell."""
    quanta = boxes["quanta"]
    volume_ids, volumes = ndimage.label(quanta > 0, structure=_CORNERS)
    sources = _two_sources(volume_ids, quanta, volumes)
    joint_ids, joint_between = _joints(quanta < 0, sources)
    joints = joint_between.shape[0]
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
    between = np.concatenate([np.full((volumes, 2), -1, dtype=np.int32), joint_between.astype(np.int32)])
    return {"owner": owner, "kind": kind, "between": between, "boxes": box_counts,
            "quanta": np.rint(amounts).astype(np.int64), "centre": centre / sizes[:, None], "clock": clocks / sizes,
            "volumes": int(volumes)}


def _best(candidates: list, empty: int) -> np.ndarray:
    """Per box the highest volume label among `candidates`, else `empty` where only empty space is among them, else 0."""
    stacked = np.stack(candidates)
    volume_best = np.where(stacked == empty, 0, stacked).max(axis=0)
    return np.where(volume_best > 0, volume_best, np.where((stacked == empty).any(axis=0), empty, 0))


def _shifted(labels: np.ndarray) -> list:
    """The six face neighbours' labels of every box (0 beyond the array)."""
    padded = np.pad(labels, 1)
    inner = labels.shape
    return [padded[1 + dx:1 + dx + inner[0], 1 + dy:1 + dy + inner[1], 1 + dz:1 + dz + inner[2]]
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]


def _two_sources(volume_ids: np.ndarray, quanta: np.ndarray, volumes: int) -> np.ndarray:
    """Per minus box the first two sources reaching it through the web, as int64 (side, side, side, 2).

    Sources are 0-based energy volumes and empty space (-2: empty boxes and the walls);
    -1 where fewer than two (or no) sources reach the box.
    """
    found = np.full(quanta.shape + (2,), -1, dtype=np.int64)
    minus = quanta < 0
    if not minus.any():
        return found
    low = np.maximum(np.argwhere(minus).min(axis=0) - 1, 0)
    high = np.minimum(np.argwhere(minus).max(axis=0) + 2, quanta.shape)
    window = tuple(slice(start, stop) for start, stop in zip(low, high))
    empty = volumes + 1
    web = np.pad(minus[window], 1)
    first = np.pad(volume_ids[window].astype(np.int64), 1)
    first[np.pad(quanta[window] == 0, 1, constant_values=True)] = empty
    second = np.zeros(first.shape, dtype=np.int64)
    while True:
        near_first = _shifted(first)
        near_second = _shifted(second)
        reached = web & (first == 0)
        new_first = np.where(reached, _best(near_first, empty), first)
        others = [np.where(near == first, 0, near) for near in near_first + near_second]
        open_second = web & (first > 0) & (second == 0)
        new_second = np.where(open_second, _best(others, empty), second)
        if np.array_equal(new_first, first) and np.array_equal(new_second, second):
            break
        first, second = new_first, new_second
    inner = (slice(1, -1),) * 3
    for slot, labels in enumerate((first[inner], second[inner])):
        source = np.where(labels == empty, -2, labels - 1)
        found[window + (slot,)] = np.where(minus[window], source, -1)
    return found


def _joints(minus: np.ndarray, sources: np.ndarray) -> tuple:
    """Joint id per box (-1 off the web) and per joint its two volumes: all minus boxes of one source pair."""
    pair = np.sort(np.where(sources == -2, -1, sources), axis=-1).reshape(-1, 2)
    flat = np.flatnonzero(minus.ravel())
    ids = np.full(minus.size, -1, dtype=np.int64)
    if flat.size == 0:
        return ids.reshape(minus.shape), np.zeros((0, 2), dtype=np.int64)
    found, first_box, joint_of = np.unique(pair[flat], axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first_box)
    renumber = np.empty(order.size, dtype=np.int64)
    renumber[order] = np.arange(order.size)
    ids[flat] = renumber[joint_of.ravel()]
    return ids.reshape(minus.shape), found[order]
