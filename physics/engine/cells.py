"""Cells: the energy volumes and joints of the forced cell lattice, read off each box's owner, on the host.

Contract: `cells(boxes)` takes host_boxes() of one universe (quanta, clock, owner,
cells_per_side; host_boxes.py) and returns a dict of numpy arrays describing its cells:
  owner     int32 (side, side, side): the cell each box belongs to, -1 for empty space
  kind      int32 per cell: ENERGY_VOLUME (0) or JOINT (1)
  between   int32 (cells, 2): for a joint the two energy volumes it lies between, lower
            id first, read from the lattice: the volumes on either side of it along an
            axis; -1 for a side that is not a volume (the outer layer faces empty
            space), (-1, -1) for a joint at an edge or corner of the volumes, and for an
            energy volume
  place     int64 (cells, 3): the cell's extended lattice indices (cell_place.py)
  boxes     int64 per cell: its box count, its size in Planck boxes
  quanta    int64 per cell: its whole quanta (amount, ignoring sign)
  centre    float64 (cells, 3): the mean index of its boxes (of its corners while it
            holds no box)
  clock     float64 per cell: the mean local clock reading of its boxes, the proper time
            it has lived (tick_clocks.py)
  volumes   int: how many energy volumes; they are cells 0 .. volumes - 1, joints after
            them, each in lattice order
The cells are the forced cell lattice's real cells (owner, 2026-11-16): a cell's kind
and its neighbours are fixed by its lattice index (cell_kind.py), so two
face-neighbouring volumes share exactly one joint, every volume is wrapped by joints on
all 26 sides and the joints are one web, whatever the boxes hold. Ids are the same at
every call. Without a cell lattice (cells_per_side 0, hand-built starts) the plus boxes
are one energy volume and the minus boxes one joint beside it.
"""
from __future__ import annotations

import numpy as np

from physics.engine.lattice_kinds import lattice_kinds

ENERGY_VOLUME = 0
JOINT = 1


def cells(boxes: dict) -> dict:
    """Owner per box and kind, the volumes each joint lies between, size, quanta, centre and mean clock per cell."""
    quanta = boxes["quanta"]
    cells_per_side = int(boxes.get("cells_per_side", 0))
    if cells_per_side == 0:
        return _signed_cells(boxes)
    kinds = lattice_kinds(cells_per_side)
    extent = cells_per_side + 2
    places = np.argwhere(np.ones(kinds.shape, dtype=bool))
    flat = kinds.ravel()
    order = np.concatenate([np.flatnonzero(flat == 1), np.flatnonzero(flat == -1)])
    renumber = np.full(flat.size, -1, dtype=np.int64)
    renumber[order] = np.arange(order.size)
    owner = renumber[boxes["owner"]].astype(np.int32)
    volumes = int((flat == 1).sum())
    between = np.full((order.size, 2), -1, dtype=np.int32)
    for row, cell in enumerate(order[volumes:], start=volumes):
        between[row] = _between(places[cell], kinds, renumber, extent)
    found = _tallies(owner, quanta, boxes["clock"], order.size)
    corner_centres = _corner_centres(boxes["corners"], places[order])
    empty = found["boxes"] == 0
    found["centre"][empty] = corner_centres[empty]
    kind = np.where(np.arange(order.size) < volumes, ENERGY_VOLUME, JOINT).astype(np.int32)
    return {"owner": owner, "kind": kind, "between": between, "place": places[order].astype(np.int64), **found,
            "volumes": volumes}


def _between(place: np.ndarray, kinds: np.ndarray, renumber: np.ndarray, extent: int) -> tuple:
    """The (new) ids of the volumes on either side of a joint along the first axis that has one, lower first."""
    for axis in range(3):
        sides = []
        for step in (-1, 1):
            near = place.copy()
            near[axis] += step
            volume = 0 <= near[axis] < extent and kinds[tuple(near)] == 1
            sides.append(int(renumber[(near[0] * extent + near[1]) * extent + near[2]]) if volume else -1)
        if max(sides) >= 0:
            return tuple(sorted(sides))
    return (-1, -1)


def _corner_centres(corners: np.ndarray, places: np.ndarray) -> np.ndarray:
    """Each cell's corner mean as a box index (minus half a box)."""
    total = np.zeros((places.shape[0], 3))
    for bits in np.ndindex(2, 2, 2):
        near = places + np.asarray(bits)
        total += corners[near[:, 0], near[:, 1], near[:, 2]]
    return total / 8.0 - 0.5


def _tallies(owner: np.ndarray, quanta: np.ndarray, clock: np.ndarray, count: int) -> dict:
    """Box count, quanta, box-index centre and mean clock per cell."""
    held = owner.ravel()
    occupied = held >= 0
    ids = held[occupied]
    box_counts = np.bincount(ids, minlength=count).astype(np.int64)
    amounts = np.bincount(ids, weights=np.abs(quanta.ravel()[occupied]).astype(np.float64), minlength=count)
    clocks = np.bincount(ids, weights=clock.ravel()[occupied].astype(np.float64), minlength=count)
    places = np.argwhere(owner >= 0)
    centre = np.stack([np.bincount(ids, weights=places[:, axis].astype(np.float64), minlength=count)
                       for axis in range(3)], axis=1)
    sizes = np.maximum(box_counts, 1)
    return {"boxes": box_counts, "quanta": np.rint(amounts).astype(np.int64), "centre": centre / sizes[:, None],
            "clock": clocks / sizes}


def _signed_cells(boxes: dict) -> dict:
    """Without a lattice: the plus boxes one energy volume, the minus boxes one joint."""
    quanta = boxes["quanta"]
    present = [sign for sign in (1, -1) if np.any(np.sign(quanta) == sign)]
    owner = np.full(quanta.shape, -1, dtype=np.int32)
    for cell, sign in enumerate(present):
        owner[np.sign(quanta) == sign] = cell
    volumes = int(1 in present)
    kind = np.array([ENERGY_VOLUME if sign > 0 else JOINT for sign in present], dtype=np.int32)
    between = np.full((len(present), 2), -1, dtype=np.int32)
    if volumes and len(present) == 2:
        between[1] = (-1, 0)
    found = _tallies(owner, quanta, boxes["clock"], len(present))
    return {"owner": owner, "kind": kind, "between": between, "place": np.zeros((len(present), 3), dtype=np.int64),
            **found, "volumes": volumes}
