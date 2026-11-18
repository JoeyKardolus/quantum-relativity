"""Cell slots: where every cell is drawn in the A B A B pattern.

Contract: `cell_slots(side, kind, length, lines=None, line_axis=None, box=None,
reach=(0, 0, 0))` takes the lattice's places per axis, per cell id its kind and its
length (the engine's size law), and what `torn_lines` returns. A cell has no place of
its own, only neighbours and a size; a drawing has to put it somewhere, and this is the
one rule for where. Every place of the start lattice is a slot and a cell never changes
slot, so nothing flickers:
  - a start cell is drawn in its own place for as long as it lives
  - the joint and the volume a tear made are drawn in the slot of the volume that tore,
    as stripes along that slot's axis, each as long as its length
  - a cell that is gone leaves its slot empty, with no width
`box` is ((x0, x1), (y0, y1), (z0, z1)), the lattice places to give cells for (None: the
whole lattice); `reach` is how many more places on each side, per axis, the widths are
wanted for (cut off at the lattice's ends). Only the cells of those places are read, so
one layer of a lattice of millions costs one layer. Returns a dict:
  piece_cell, piece_slot   one row per living cell of the box: its id and its slot
  piece_low, piece_high    the share of the slot it takes along its stripe axis (0 and
                           1 for a cell alone in its slot)
  slot_axis   {slot: axis} the axis a torn slot's stripes lie along
  first       the lattice place of the first slot the widths are for
  widths      (3, nx, ny, nz) every slot's width (`slot_widths`) over the box and its
              reach; a shared slot is the sum of its cells' lengths along its axis and
              its volumes' mean across
Lengths are those of `drawn_length`, taken over the places read.
"""
from __future__ import annotations

import numpy as np

from physics.view.layout.definitions.drawn_length import drawn_length
from physics.view.layout.definitions.kinds import ENERGY_VOLUME, NO_CELL
from physics.view.layout.definitions.slot_widths import slot_widths


def cell_slots(side: int, kind: np.ndarray, length: np.ndarray, lines: dict | None = None,
               line_axis: dict | None = None, box: tuple | None = None, reach: tuple = (0, 0, 0)) -> dict:
    """Give every living cell of the box its slot and every slot round it its widths."""
    box = box or ((0, side),) * 3
    kind, length = np.asarray(kind), np.asarray(length)
    lines, line_axis = lines or {}, line_axis or {}
    wide = [(max(low - far, 0), min(high + far, side)) for (low, high), far in zip(box, reach)]
    read = [np.arange(low - 1, high + 1) % side for low, high in wide]
    slot = (read[0][:, None, None] * side + read[1][None, :, None]) * side + read[2][None, None, :]
    alive = kind[slot] != NO_CELL
    along = drawn_length(length[slot], alive)
    longest = float(along.max()) if along.size and along.max() > 0.0 else 1.0
    across = along.copy()
    slot_axis = np.zeros(slot.shape, dtype=np.int8)
    inner = tuple(slice(1 + low - far_low, 1 + high - far_low) for (low, high), (far_low, _) in zip(box, wide))
    single = np.zeros(slot.shape, dtype=bool)
    single[inner] = alive[inner]
    cells, slots, lows, highs = [], [], [], []
    for torn, living in lines.items():
        place = (torn // (side * side), (torn // side) % side, torn % side)
        seen = np.ix_(*[np.flatnonzero(read[axis] == place[axis]) for axis in range(3)])
        if not slot[seen].size:
            continue
        sizes = length[living].astype(np.float64)
        sizes = np.where(np.isfinite(sizes), sizes, longest)
        volumes = sizes[kind[living] == ENERGY_VOLUME]
        total = float(sizes.sum())
        along[seen], slot_axis[seen], single[seen] = total, line_axis[torn], False
        across[seen] = float((volumes if len(volumes) else sizes).mean()) if len(living) else 0.0
        if all(low <= at < high for at, (low, high) in zip(place, box)):
            edges = np.concatenate([[0.0], np.cumsum(sizes)]) / max(total, 1e-30)
            cells.append(np.array(living, dtype=np.int64))
            slots.append(np.full(len(living), torn, dtype=np.int64))
            lows.append(edges[:-1])
            highs.append(edges[1:])
    alone = slot[single].astype(np.int64)
    return {"piece_cell": np.concatenate([alone, *cells]), "piece_slot": np.concatenate([alone, *slots]),
            "piece_low": np.concatenate([np.zeros(len(alone)), *lows]),
            "piece_high": np.concatenate([np.ones(len(alone)), *highs]), "slot_axis": dict(line_axis),
            "first": tuple(low for low, _ in wide),
            "widths": slot_widths(along, across, slot_axis, tuple(low for low, _ in wide))}
