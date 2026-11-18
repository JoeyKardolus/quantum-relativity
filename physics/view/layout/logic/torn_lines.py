"""Torn lines: which cells share a slot after tears, and in what order.

Contract: `torn_lines(lattice, kind, torn_from, made_kind, swirl_axis)` reads four of
the engine's per-cell arrays (one row per id handed out; the first `lattice` ids are
the start's cells). A tear of volume V makes a joint J and a volume W, the next two ids,
both with `torn_from` = V and V's swirl axis; V keeps the low piece. Returns
`(lines, line_axis)`:
  lines      {slot: [ids]} for every slot a tear touched: its living cells in the order
             they lie along the tear, V first, J and W right after the volume they were
             torn from (a later tear of V goes between V and the earlier J)
  line_axis  {slot: axis} the axis that slot's cells are striped along: the swirl axis
             of the first tear there
A slot is the lattice place of the start volume the line began with. Raises ValueError
when an id past the lattice is not half of such a pair: the engine then makes cells in
a way this drawing does not know.
"""
from __future__ import annotations

import numpy as np

from physics.view.layout.definitions.kinds import ENERGY_VOLUME, JOINT, NO_CELL


def torn_lines(lattice: int, kind: np.ndarray, torn_from: np.ndarray, made_kind: np.ndarray,
               swirl_axis: np.ndarray) -> tuple:
    """Every torn slot's living cells in order, and the axis they are striped along."""
    count = len(kind)
    if (count - lattice) % 2:
        raise ValueError(f"{count - lattice} ids past the lattice: tears make cells in pairs")
    order, root, line_axis = {}, {}, {}
    for joint in range(lattice, count, 2):
        volume = int(torn_from[joint])
        if (volume < 0 or int(torn_from[joint + 1]) != volume or int(made_kind[joint]) != JOINT
                or int(made_kind[joint + 1]) != ENERGY_VOLUME):
            raise ValueError(f"ids {joint} and {joint + 1} are not the joint and volume of one tear")
        slot = root.get(volume, volume)
        if slot >= lattice:
            raise ValueError(f"cell {joint} was torn from {volume}, which has no slot")
        root[joint] = root[joint + 1] = slot
        if slot not in order:
            order[slot] = [slot]
            line_axis[slot] = int(swirl_axis[joint])
        at = order[slot].index(volume) + 1
        order[slot][at:at] = [joint, joint + 1]
    lines = {slot: [cell for cell in line if kind[cell] != NO_CELL] for slot, line in order.items()}
    return lines, line_axis
