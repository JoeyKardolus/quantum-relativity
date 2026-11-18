"""Held share: lumps that hold, the energy in lumps two thirds through that is still in lumps at the end.

Contract: `held_share(snapshots, fell_into)` takes `take_snapshot` dicts in time order
and the grid's `fell_into` (per cell id: the volume a vanished volume fell into, -1 for
none) and returns a share between 0 and 1: of the plus quanta at the snapshot two thirds
through, the share that sat in lump volumes then whose cell (or the cell it has since
fallen into, followed to the end) is a lump volume in the last snapshot. 0.0 when either
snapshot has no lump.
"""
from __future__ import annotations


def _where_now(cell: int, fell_into) -> int:
    """The cell a cell's energy is in now: itself, or the volume it fell into, followed to the end."""
    while fell_into[cell] >= 0:
        cell = int(fell_into[cell])
    return cell


def held_share(snapshots: list, fell_into) -> float:
    """The share of the plus quanta two thirds through that was in lumps then and is in lumps at the end."""
    then, last = snapshots[(2 * (len(snapshots) - 1)) // 3], snapshots[-1]
    if len(then["lumps"]) == 0 or len(last["lumps"]) == 0:
        return 0.0
    still = set(last["lumps"].tolist())
    kept = sum(int(then["held"][cell]) for cell in then["lumps"].tolist() if _where_now(cell, fell_into) in still)
    return float(kept / then["plus"])
