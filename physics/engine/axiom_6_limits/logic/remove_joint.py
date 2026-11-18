"""Remove joint: a joint that goes takes its energy with it.

Contract: `remove_joint(state, held, joint)` takes one joint out of the grid: its quanta,
counted from its boxes, are booked in `taken_joint` and its boxes become 0; its id is no
cell; every neighbour drops its entries for it. Returns the quanta it held. It merges
nothing and gives no other cell a new relation.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_3_structure.definitions.kinds import BLOCK, NO_CELL
from physics.engine.axiom_6_limits.logic.hold_cell import hold_cell
from physics.engine.axiom_6_limits.logic.set_value import set_value


def remove_joint(state, held, joint: int) -> int:
    """Take the joint out of the grid; returns the quanta it held."""
    hold_cell(state, held, joint, with_block=True)
    entries = list(held.entries[joint])
    for neighbour in sorted({entry[0] for entry in entries}):
        hold_cell(state, held, neighbour)
        held.entries[neighbour] = [entry for entry in held.entries[neighbour] if entry[0] != joint]
    taken = int(np.abs(held.quanta[joint].astype(np.int64)).sum())
    held.quanta[joint] = np.zeros((BLOCK, BLOCK, BLOCK), dtype=np.int32)
    held.phase[joint] = np.zeros((BLOCK, BLOCK, BLOCK), dtype=np.float64)
    held.entries[joint] = []
    set_value(state, held, "kind", joint, NO_CELL)
    set_value(state, held, "heat_quanta", joint, 0)
    state.taken_joint += taken
    return taken
