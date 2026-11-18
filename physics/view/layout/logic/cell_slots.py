"""Cell slots: where every cell is drawn in the A B A B pattern.

Contract: `cell_slots(side, kind, length, lines=None, line_axis=None)` takes the
lattice's places per axis, per cell id its kind and its length (the engine's size law),
and what `torn_lines` returns. A cell has no place of its own, only neighbours and a
size; a drawing has to put it somewhere, and this is the one rule for where. Every
place of the start lattice is a slot and a cell never changes slot, so nothing flickers:
  - a start cell is drawn in its own place for as long as it lives
  - the joint and the volume a tear made are drawn in the slot of the volume that tore,
    as stripes along that slot's axis, each as long as its length
  - a cell that is gone leaves its slot empty, with no width: the grid closes up there
Returns a dict of numpy arrays:
  piece_cell, piece_slot   one row per living cell: its id and its slot
  piece_low, piece_high    the share of the slot it takes along slot_axis (0 and 1 for
                           a cell alone in its slot)
  slot_axis   (side ** 3,) the axis a slot's stripes lie along (0 where none tore)
  widths      (3, side, side, side) every slot's width (`slot_widths`); a shared slot is
              the sum of its cells' lengths along its axis and its volumes' mean across
Lengths are those of `drawn_length`.
"""
from __future__ import annotations

import numpy as np

from physics.view.layout.definitions.drawn_length import drawn_length
from physics.view.layout.definitions.kinds import ENERGY_VOLUME, NO_CELL
from physics.view.layout.definitions.slot_widths import slot_widths


def cell_slots(side: int, kind: np.ndarray, length: np.ndarray, lines: dict | None = None,
               line_axis: dict | None = None) -> dict:
    """Give every living cell its slot and every slot its widths."""
    lattice = side ** 3
    kind = np.asarray(kind)
    length = drawn_length(length, kind != NO_CELL)
    lines, line_axis = lines or {}, line_axis or {}
    slot_axis = np.zeros(lattice, dtype=np.int64)
    along = length[:lattice].copy()
    across = along.copy()
    single = kind[:lattice] != NO_CELL
    single[list(lines)] = False
    alone = np.flatnonzero(single)
    cells, slots, lows, highs = [alone], [alone], [np.zeros(len(alone))], [np.ones(len(alone))]
    for slot, living in lines.items():
        slot_axis[slot] = line_axis[slot]
        sizes = length[living]
        volumes = sizes[kind[living] == ENERGY_VOLUME]
        along[slot] = sizes.sum()
        across[slot] = (volumes if len(volumes) else sizes).mean() if len(living) else 0.0
        edges = np.concatenate([[0.0], np.cumsum(sizes)]) / max(along[slot], 1e-30)
        cells.append(np.array(living, dtype=np.int64))
        slots.append(np.full(len(living), slot, dtype=np.int64))
        lows.append(edges[:-1])
        highs.append(edges[1:])
    return {"piece_cell": np.concatenate(cells), "piece_slot": np.concatenate(slots),
            "piece_low": np.concatenate(lows), "piece_high": np.concatenate(highs), "slot_axis": slot_axis,
            "widths": slot_widths(side, along, across, slot_axis)}
