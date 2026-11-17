"""Cell slots: where every cell is drawn in the A B A B pattern, and how wide each place of the pattern is.

Contract: `cell_slots(side, kind, length, torn_from=None)` takes the lattice's places per
axis, per cell id its kind (1 energy volume, -1 joint, 0 no cell) and its length (the
size law's, physics.engine), and the engine's record of tears {new id: (volume torn,
axis)}. A cell has no place of its own, only neighbours and a size; a drawing has to put
it somewhere, and this is the one rule for where. Every place of the lattice is a slot,
and a cell never changes slot, so nothing can flicker:
  - a lattice cell is drawn in its own place for as long as it lives
  - the joint and the volume a tear made are drawn in the slot of the volume that tore,
    right after it, as stripes along the first axis that slot tore along
  - a cell that is gone (a collapsed joint, a volume that fell into another) leaves its
    slot empty, with no width: the grid closes up there
Returns a dict of numpy arrays:
  piece_cell, piece_slot   one row per living cell: its id and the lattice place it is
                    drawn in; piece_low, piece_high the share of the slot it takes along
                    slot_axis (0 and 1 for a cell alone in its slot)
  slot_axis         (side ** 3,) the axis a slot's stripes lie along (0 where none tore)
  widths            (3, side, side, side) float64 how wide every slot is along x, y, z:
                    along an axis where the cell is itself (every axis of an energy
                    volume, the odd axes of a joint: a face joint is thin one way, an
                    edge joint two, a corner joint three) its own length (several cells
                    in one slot: their lengths summed along slot_axis, the volumes' mean
                    across); along an axis where a joint only lies beside volumes, the
                    mean width of those volumes, so mortar is as long as its bricks
"""
from __future__ import annotations

import numpy as np


def cell_slots(side: int, kind: np.ndarray, length: np.ndarray, torn_from: dict | None = None) -> dict:
    """Give every living cell its slot and every slot its widths."""
    lattice = side ** 3
    kind, length = np.asarray(kind), np.asarray(length, dtype=np.float64)
    alive = kind[:lattice] != 0
    slot_axis = np.zeros(lattice, dtype=np.int64)
    along = np.where(alive, length[:lattice], 0.0)
    across = along.copy()
    shared = _shared(lattice, kind, torn_from or {}, slot_axis)
    single = alive.copy()
    single[list(shared)] = False
    cells, slots = [np.flatnonzero(single)], [np.flatnonzero(single)]
    lows, highs = [np.zeros(int(single.sum()))], [np.ones(int(single.sum()))]
    for slot, living in shared.items():
        sizes = length[living]
        volumes = sizes[kind[living] == 1]
        along[slot] = sizes.sum()
        across[slot] = (volumes if len(volumes) else sizes).mean() if len(living) else 0.0
        edges = np.concatenate([[0.0], np.cumsum(sizes)]) / max(along[slot], 1e-30)
        cells.append(np.array(living, dtype=np.int64))
        slots.append(np.full(len(living), slot, dtype=np.int64))
        lows.append(edges[:-1])
        highs.append(edges[1:])
    return {"piece_cell": np.concatenate(cells), "piece_slot": np.concatenate(slots),
            "piece_low": np.concatenate(lows), "piece_high": np.concatenate(highs), "slot_axis": slot_axis,
            "widths": _widths(side, along, across, slot_axis)}


def _shared(lattice: int, kind: np.ndarray, torn_from: dict, slot_axis: np.ndarray) -> dict:
    """Slots a tear touched: {slot: its living cells in drawing order}; sets those slots' stripe axis."""
    order, root = {}, {}
    for joint in sorted(torn_from):
        if (joint - lattice) % 2:
            continue
        volume, axis = torn_from[joint]
        slot = root.get(volume, volume)
        root[joint] = root[joint + 1] = slot
        if slot not in order:
            order[slot] = [slot]
            slot_axis[slot] = axis
        at = order[slot].index(volume) + 1
        order[slot][at:at] = [joint, joint + 1]
    return {slot: [cell for cell in line if kind[cell] != 0] for slot, line in order.items()}


def _widths(side: int, along: np.ndarray, across: np.ndarray, slot_axis: np.ndarray) -> np.ndarray:
    """Every slot's width along each axis: its own where the cell is itself, its volumes' mean where it lies beside them."""
    index = np.indices((side, side, side))
    odd = index % 2 == 1
    widths = np.zeros((3, side, side, side))
    for axis in range(3):
        own = np.where(slot_axis == axis, along, across).reshape(side, side, side)
        wide = own.copy()
        for other in range(3):
            if other != axis:
                beside = 0.5 * (np.roll(wide, 1, axis=other) + np.roll(wide, -1, axis=other))
                wide = np.where(odd[other] & ~odd[axis], beside, wide)
        widths[axis] = wide
    return widths
