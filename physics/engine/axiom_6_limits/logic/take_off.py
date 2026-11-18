"""Take off: a collapsed joint's energy is taken off the rest.

Contract: `take_off(state, held, volume, wanted)` takes `wanted` plus quanta (the number
the gone joint held) off the volume's boxes, in proportion to what each holds
(`whole_shares`), exactly. When the volume holds less than that, all it holds is taken,
`short_takes` rises by 1 and `short_quanta` by what could not be taken. The same share
of the volume's heat goes with it (rounded up, so the volume never holds more heat than
quanta). Returns what was really taken, booked in `taken_volumes`.
"""
from __future__ import annotations

import numpy as np

from physics.engine.axiom_1_energy.definitions.whole_shares import whole_shares
from physics.engine.axiom_3_structure.definitions.kinds import BLOCK
from physics.engine.axiom_6_limits.logic.cell_value import cell_value
from physics.engine.axiom_6_limits.logic.hold_cell import hold_cell
from physics.engine.axiom_6_limits.logic.set_value import set_value


def take_off(state, held, volume: int, wanted: int) -> int:
    """Take the joint's number of quanta off the volume; returns what was really taken."""
    hold_cell(state, held, volume, with_block=True)
    holds = [int(value) for value in held.quanta[volume].ravel()]
    before = sum(holds)
    taken = min(int(wanted), before)
    if taken < wanted:
        state.short_takes += 1
        state.short_quanta += int(wanted) - taken
    shares = np.array(whole_shares(taken, holds), dtype=np.int64).reshape(BLOCK, BLOCK, BLOCK)
    held.quanta[volume] = (held.quanta[volume].astype(np.int64) - shares).astype(np.int32)
    if before > 0:
        heat = int(cell_value(state, held, "heat_quanta", volume))
        set_value(state, held, "heat_quanta", volume, heat - (taken * heat + before - 1) // before)
    state.taken_volumes += taken
    return taken
